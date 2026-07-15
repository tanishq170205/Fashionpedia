"""
LLM-based query parser.

Converts a raw natural language query into structured fields that the reranker
can operate on. Uses Groq (llama-3.3-70b-versatile) via the groq Python SDK.

Why LLM and not regex or keyword matching:
  Fashion queries are phrased in an enormous variety of ways. "Something to
  wear to a job interview" and "formal business attire" describe nearly the same
  thing. A regex pattern for "formal" would catch the second but not the first.
  An LLM that has read enough English can make that inference. The structured
  output contract is enforced by the prompt; the LLM is not being asked to do
  reasoning, just extraction, which models of this size do reliably.

Fallback behavior:
  If the Groq call fails (no API key, network error, rate limit) or returns
  malformed JSON, the parser returns {"garments": [], "setting": None} and logs
  a warning. The pipeline degrades gracefully: stage-1 CLIP similarity still
  runs, and the reranker returns all-zero attribute scores. The final ranking is
  stage-1-only, which is still better than nothing.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedQuery:
    # Each garment dict has {"label": str, "color": str | None}.
    garments: list[dict] = field(default_factory=list)
    # Setting is a short description of scene/context, or None if not present.
    setting: Optional[str] = None
    # True if the LLM call succeeded; False means we're running on keyword fallback.
    llm_succeeded: bool = True


_SYSTEM_PROMPT = """You are a fashion query parser. Given a natural language query about clothing or outfits, extract structured information.

Return ONLY a JSON object with this exact schema, no explanation, no markdown, no extra fields:
{"garments": [{"label": "...", "color": "..."}], "setting": "..."}

Rules:
- "garments" is a list of garment+color pairs. Each item must have "label" (garment type, e.g. "shirt", "pants", "dress") and "color" (color word, e.g. "red", "navy", "burgundy"). Set "color" to null if no color is mentioned for that garment.
- "setting" is a short phrase describing the scene, occasion, or context (e.g. "office", "park", "formal occasion", "casual outdoor"). Set to null if no context is implied.
- If no garments are mentioned at all (just vibe/setting), return an empty garments list.
- Use simple English color words, not hex codes or RGB.
- Do not invent garments that are not implied by the query.

Examples:
Query: "a red tie and a white shirt in a formal setting"
Output: {"garments": [{"label": "tie", "color": "red"}, {"label": "shirt", "color": "white"}], "setting": "formal occasion"}

Query: "casual weekend outfit for a city walk"
Output: {"garments": [], "setting": "casual outdoor city"}

Query: "someone wearing a blue shirt sitting on a park bench"
Output: {"garments": [{"label": "shirt", "color": "blue"}], "setting": "park outdoor"}"""


def parse_query(query: str, groq_model: str = "llama-3.3-70b-versatile") -> ParsedQuery:
    """
    Parse a natural language query into structured fields.

    Attempts Groq API first; falls back to keyword extraction on any failure.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        warnings.warn(
            "GROQ_API_KEY not set. Falling back to keyword extraction. "
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
            temperature=0.0,   # deterministic — we want extraction, not creativity
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps the JSON anyway.
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        return _validate_and_build(parsed, llm_succeeded=True)

    except json.JSONDecodeError as e:
        warnings.warn(
            f"LLM returned malformed JSON: {e}. Falling back to keyword extraction.",
            stacklevel=2,
        )
        return _keyword_fallback(query)

    except Exception as e:
        warnings.warn(
            f"Groq API call failed: {e}. Falling back to keyword extraction.",
            stacklevel=2,
        )
        return _keyword_fallback(query)


def _validate_and_build(parsed: dict, llm_succeeded: bool) -> ParsedQuery:
    """
    Validate the LLM output structure and return a ParsedQuery.
    Coerces types rather than raising on minor schema deviations.
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
        garments.append({"label": label.strip().lower(), "color": color})

    setting = parsed.get("setting")
    if not isinstance(setting, str) or not setting.strip():
        setting = None

    return ParsedQuery(garments=garments, setting=setting, llm_succeeded=llm_succeeded)


def _keyword_fallback(query: str) -> ParsedQuery:
    """
    Extract garment+color pairs using simple token-matching when the LLM is
    unavailable.

    This is deliberately minimal — it catches the obvious patterns ("blue shirt",
    "red tie") but misses paraphrases ("something formal", "business casual").
    The comment here is intentional: this fallback should be visible in the code
    as a degraded path, not presented as equivalent to the LLM parser.
    """
    COLORS = {
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
        "orange", "purple", "pink", "brown", "beige", "navy", "teal", "maroon",
        "burgundy", "olive", "cream", "ivory", "gold", "silver", "coral",
        "turquoise", "indigo", "lavender", "rust", "camel", "khaki",
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
        "office": "office indoor formal",
        "work": "office indoor formal",
        "formal": "formal occasion",
        "business": "office indoor formal",
        "park": "park outdoor",
        "outdoor": "outdoor",
        "indoor": "indoor",
        "casual": "casual outdoor",
        "beach": "beach outdoor",
        "wedding": "formal occasion",
        "party": "social event",
        "gym": "athletic outdoor",
        "street": "street outdoor urban",
        "city": "urban outdoor",
    }

    tokens = query.lower().split()
    garments = []
    setting = None

    # Simple bigram scan: if a color precedes a garment, pair them.
    for i, token in enumerate(tokens):
        clean = token.strip(".,!?")
        if clean in GARMENTS:
            color = None
            if i > 0 and tokens[i - 1].strip(".,!?") in COLORS:
                color = tokens[i - 1].strip(".,!?")
            garments.append({"label": clean, "color": color})

    # Setting: take the first matched keyword.
    for token in tokens:
        clean = token.strip(".,!?")
        if clean in SETTINGS:
            setting = SETTINGS[clean]
            break

    return ParsedQuery(garments=garments, setting=setting, llm_succeeded=False)
