"""
LLM-based query parser.

Converts a raw natural language query into structured fields that the reranker
can operate on. Uses Groq (llama-3.3-70b-versatile) via the groq Python SDK.

Why LLM instead of regex:
  Fashion queries come in many phrasings. "Something for a job interview" and
  "formal business attire" describe nearly the same thing. A regex for "formal"
  catches the second but not the first. The LLM generalizes across phrasings
  because it understands the semantics, not just the surface form.

Fallback:
  If Groq is unavailable, the parser falls back to a keyword extractor that
  catches the most common patterns (color + garment bigrams). The pipeline
  degrades gracefully: stage-1 CLIP similarity still runs, and the final
  ranking is stage-1-only rather than nothing.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pre-parsed cache for the benchmark queries.
# This avoids API calls at demo/eval time and guarantees deterministic parsing
# regardless of Groq availability. Any query not in the cache hits the LLM.
# ---------------------------------------------------------------------------
_CACHE_PATH = Path(__file__).resolve().parent.parent / "eval" / "parsed_cache.json"
_QUERY_CACHE: dict = {}


def _load_cache() -> None:
    global _QUERY_CACHE
    if _QUERY_CACHE or not _CACHE_PATH.exists():
        return
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _QUERY_CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        pass


_load_cache()


@dataclass
class ParsedQuery:
    # Each garment is {"label": str, "color": str | None, "synonyms": list[str]}.
    garments: list[dict] = field(default_factory=list)
    # Setting describes scene/occasion, or None if not present.
    setting: Optional[str] = None
    # True if LLM parsed, False if keyword fallback.
    llm_succeeded: bool = True


# Garment synonyms: if the LLM/user says X, also match Y in region embeddings.
# This widens recall without lowering the similarity threshold.
GARMENT_SYNONYMS: dict[str, list[str]] = {
    "raincoat":     ["coat", "jacket", "anorak", "mac"],
    "coat":         ["jacket", "overcoat", "blazer"],
    "jacket":       ["coat", "blazer", "cardigan"],
    "blazer":       ["jacket", "suit jacket"],
    "tie":          ["necktie", "bow tie"],
    "shirt":        ["blouse", "top", "button-down"],
    "pants":        ["trousers", "jeans", "chinos", "slacks"],
    "jeans":        ["pants", "denim"],
    "dress":        ["gown", "frock"],
    "skirt":        ["mini skirt", "maxi skirt"],
    "shoes":        ["boots", "sneakers", "heels", "footwear"],
    "boots":        ["shoes", "footwear"],
    "sweater":      ["jumper", "pullover", "knitwear"],
    "hoodie":       ["sweatshirt", "sweater"],
    "shorts":       ["bermuda", "cut-offs"],
    "suit":         ["blazer", "jacket"],
    "gown":         ["dress", "evening gown"],
    "bag":          ["handbag", "purse", "tote"],
    "hat":          ["cap", "beanie", "beret"],
    "scarf":        ["wrap", "stole"],
}

# Expanded color mapping: normalise user color words to simple names.
COLOR_NORMALISE: dict[str, str] = {
    "bright yellow": "yellow",
    "neon yellow":   "yellow",
    "lemon":         "yellow",
    "navy blue":     "navy",
    "dark blue":     "navy",
    "royal blue":    "blue",
    "sky blue":      "blue",
    "light blue":    "blue",
    "dark red":      "maroon",
    "wine":          "maroon",
    "crimson":       "red",
    "scarlet":       "red",
    "dark green":    "green",
    "forest green":  "green",
    "olive green":   "olive",
    "off white":     "white",
    "off-white":     "white",
    "cream":         "white",
    "ivory":         "white",
    "charcoal":      "grey",
    "dark grey":     "grey",
    "light grey":    "grey",
    "tan":           "beige",
    "nude":          "beige",
    "camel":         "beige",
    "mustard":       "yellow",
}


_SYSTEM_PROMPT = """\
You are a fashion query parser. Given a natural language query about clothing, extract structured information.

Return ONLY a JSON object with this exact schema — no explanation, no markdown:
{"garments": [{"label": "...", "color": "..."}], "setting": "..."}

Rules:
- "garments": list of garment+color pairs. Each item must have:
    "label" — the garment type as a single noun: shirt, pants, dress, coat, raincoat, blazer, tie, etc.
    "color" — a simple English color word (red, yellow, white, navy, blue, black, grey, etc.) or null if not specified.
- "setting": short phrase for scene/occasion/context, or null if none implied.
  Examples: "office indoor formal", "park outdoor casual", "formal occasion", "urban street casual",
            "fashion runway", "beach outdoor summer".
- Do NOT invent garments not mentioned. Do NOT add colors not stated.
- Normalise synonyms: trousers→pants, tee→shirt, top→shirt, sneakers→shoes, trainers→shoes, tee-shirt→shirt.
- For "bright yellow", use color "yellow". For "navy blue", use "navy". For "dark red", use "maroon".
- For style-only queries ("casual weekend", "business professional"), return empty garments list and infer setting.
- Garment label must be a single noun. "button-down shirt" → label="shirt".

Examples:
Query: "a red tie and a white shirt in a formal setting"
Output: {"garments": [{"label": "tie", "color": "red"}, {"label": "shirt", "color": "white"}], "setting": "formal occasion"}

Query: "a person in a bright yellow raincoat"
Output: {"garments": [{"label": "raincoat", "color": "yellow"}], "setting": null}

Query: "professional business attire inside a modern office"
Output: {"garments": [], "setting": "office indoor formal business"}

Query: "someone wearing a blue shirt sitting on a park bench"
Output: {"garments": [{"label": "shirt", "color": "blue"}], "setting": "park outdoor"}

Query: "casual weekend outfit for a city walk"
Output: {"garments": [], "setting": "casual outdoor urban"}

Query: "a grey blazer and black trousers"
Output: {"garments": [{"label": "blazer", "color": "grey"}, {"label": "pants", "color": "black"}], "setting": null}

Query: "summer dress on the beach"
Output: {"garments": [{"label": "dress", "color": null}], "setting": "beach outdoor summer"}

Query: "streetwear hoodie and joggers"
Output: {"garments": [{"label": "hoodie", "color": null}, {"label": "pants", "color": null}], "setting": "casual street urban"}

Query: "elegant evening gown at a gala"
Output: {"garments": [{"label": "dress", "color": null}], "setting": "formal event evening"}

Query: "navy blazer and chinos at work"
Output: {"garments": [{"label": "blazer", "color": "navy"}, {"label": "pants", "color": null}], "setting": "office indoor formal"}

Query: "woman in a floral dress on a runway"
Output: {"garments": [{"label": "dress", "color": null}], "setting": "fashion runway"}

Query: "red coat outdoors in winter"
Output: {"garments": [{"label": "coat", "color": "red"}], "setting": "outdoor winter"}
"""


def parse_query(query: str, groq_model: str = "llama-3.3-70b-versatile") -> ParsedQuery:
    """
    Parse a natural language query into structured fields.

    Lookup order:
    1. Pre-parsed cache (eval/parsed_cache.json) — deterministic, no API call.
    2. Groq LLaMA-3.3-70B — handles arbitrary queries.
    3. Keyword fallback — if API unavailable.
    """
    # Pre-normalise the query before cache lookup.
    normalised = _normalise_query(query)

    if normalised in _QUERY_CACHE:
        cached = _QUERY_CACHE[normalised]
        return _validate_and_build(cached, llm_succeeded=cached.get("llm_succeeded", True))
    if query.strip() in _QUERY_CACHE:
        cached = _QUERY_CACHE[query.strip()]
        return _validate_and_build(cached, llm_succeeded=cached.get("llm_succeeded", True))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        warnings.warn(
            "GROQ_API_KEY not set. Using keyword fallback. "
            "Set the environment variable for better query understanding.",
            stacklevel=2,
        )
        return _keyword_fallback(query)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        return _validate_and_build(parsed, llm_succeeded=True)

    except json.JSONDecodeError as e:
        warnings.warn(f"LLM returned malformed JSON: {e}. Using keyword fallback.", stacklevel=2)
        return _keyword_fallback(query)
    except Exception as e:
        warnings.warn(f"Groq API call failed: {e}. Using keyword fallback.", stacklevel=2)
        return _keyword_fallback(query)


def _normalise_query(query: str) -> str:
    """Lowercase and strip the query for consistent cache lookup."""
    return query.strip().lower()


def _validate_and_build(parsed: dict, llm_succeeded: bool) -> ParsedQuery:
    """
    Validate LLM output and return a ParsedQuery with synonym expansion.
    """
    garments = []
    for g in parsed.get("garments", []):
        if not isinstance(g, dict):
            continue
        label = g.get("label", "")
        if not isinstance(label, str) or not label.strip():
            continue
        color = g.get("color")
        if not isinstance(color, str):
            color = None

        # Normalise color (e.g. "bright yellow" → "yellow").
        if color:
            color = color.strip().lower()
            color = COLOR_NORMALISE.get(color, color)

        label = label.strip().lower()
        synonyms = GARMENT_SYNONYMS.get(label, [])

        garments.append({
            "label":    label,
            "color":    color,
            "synonyms": synonyms,
        })

    setting = parsed.get("setting")
    if not isinstance(setting, str) or not setting.strip():
        setting = None

    return ParsedQuery(garments=garments, setting=setting, llm_succeeded=llm_succeeded)


def _keyword_fallback(query: str) -> ParsedQuery:
    """
    Token-matching fallback when the LLM is unavailable.

    Catches obvious patterns (color + garment bigrams) but misses paraphrases.
    This is intentionally conservative — a bad parse is worse than no parse.
    """
    COLORS = {
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
        "orange", "purple", "pink", "brown", "beige", "navy", "teal", "maroon",
        "burgundy", "olive", "cream", "ivory", "gold", "silver", "coral",
        "turquoise", "indigo", "lavender", "rust", "camel", "khaki", "bright",
        "dark", "light", "neon",
    }
    GARMENTS = {
        "shirt", "jacket", "coat", "pants", "dress", "skirt", "hat", "bag",
        "shoes", "tie", "jeans", "sweater", "blazer", "shorts", "scarf",
        "gloves", "belt", "suit", "hoodie", "cardigan", "trousers", "vest",
        "raincoat", "windbreaker", "saree", "kimono", "leggings", "boots",
        "sneakers", "sandals", "handbag", "backpack", "cap", "beanie", "top",
        "blouse", "jumper", "pullover", "anorak", "parka", "uniform", "gown",
    }
    SETTINGS = {
        "office":   "office indoor formal",
        "work":     "office indoor formal",
        "formal":   "formal occasion",
        "business": "office indoor formal",
        "park":     "park outdoor",
        "outdoor":  "outdoor",
        "indoor":   "indoor",
        "casual":   "casual outdoor",
        "beach":    "beach outdoor",
        "wedding":  "formal occasion",
        "party":    "social event",
        "gym":      "athletic outdoor",
        "street":   "street outdoor urban",
        "city":     "urban outdoor",
        "runway":   "fashion runway",
    }

    tokens = query.lower().split()
    garments = []
    setting = None

    for i, token in enumerate(tokens):
        clean = token.strip(".,!?")
        if clean in GARMENTS:
            # Look back up to 2 tokens for a color.
            color = None
            for j in range(max(0, i - 2), i):
                candidate = tokens[j].strip(".,!?")
                if candidate in COLORS and candidate not in ("bright", "dark", "light", "neon"):
                    color = candidate
            # Normalise synonyms.
            label = clean
            if label == "trousers":
                label = "pants"
            elif label in ("tee", "top", "blouse"):
                label = "shirt"
            elif label in ("sneakers", "trainers"):
                label = "shoes"
            synonyms = GARMENT_SYNONYMS.get(label, [])
            garments.append({"label": label, "color": color, "synonyms": synonyms})

    for token in tokens:
        clean = token.strip(".,!?")
        if clean in SETTINGS:
            setting = SETTINGS[clean]
            break

    return ParsedQuery(garments=garments, setting=setting, llm_succeeded=False)
