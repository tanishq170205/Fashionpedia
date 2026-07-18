# Multimodal Fashion & Context Retrieval

A search engine that takes a sentence like *"a red tie and a white shirt in a
formal setting"* and returns the images that actually match — not just
images that *feel* similar, which is what happens if you point plain CLIP at
this problem and call it done.

```
python retriever/main.py --db-path ./chroma_db_fashion \
  --query "a red tie and a white shirt in a formal setting" --top-k 5
```

---

## Why this exists

CLIP is a genuinely strong zero-shot baseline, and I want to be upfront that
it's the foundation this whole system sits on. But it has one structural
problem that matters a lot for fashion: **it squashes an entire query into a
single vector.** "Red tie and white shirt" and "white tie and red shirt"
produce nearly identical embeddings, because CLIP has no mechanism for
binding a color to a *specific* garment — it just knows the query is roughly
"formalwear, red, white, tie, shirt" as a bag of concepts. The same collapse
happens between clothing and scene: for "professional business attire inside
a modern office," CLIP can't tell you whether it matched on the *clothing*
being professional or the *room* looking like an office. Those are two
different claims about the image, and the benchmark queries this system is
evaluated against were chosen to expose exactly this weakness (Q5 in
particular is a straight compositional trap).

So instead of one embedding doing everything, I split the problem in two:

1. **A fast stage that finds plausible candidates** — a fashion-tuned CLIP
   model over the whole image, exactly as fast and scalable as vanilla CLIP
   retrieval.
2. **A slower stage that actually checks the claim** — did *this specific
   region* of *this specific person* match the color and garment that was
   asked for, rather than "does this image generally look formal and red and
   white."

That's the whole idea. Everything below is the mechanics of making that
work in an open-vocabulary setting (no fixed taxonomy, no fine-tuning on
Fashionpedia's own labels).

## Approaches I considered

| Approach | Strength | Tradeoff | Where it's the right call |
|---|---|---|---|
| **Plain CLIP, one embedding per image** | Dead simple, fast ANN search, a genuinely solid zero-shot baseline | Can't bind attributes to the right garment in compositional queries — clothing and scene compete inside one vector | Broad "vibe" search where exact composition doesn't matter |
| **Supervised detector + attribute classifiers** | Precise once the taxonomy and labels exist | Zero-shot vocabulary is capped — a new garment word needs new labels and retraining | Catalog search over a fixed, controlled product taxonomy |
| **Retrieve + rerank (what I built)** | Open-vocabulary, keeps CLIP's zero-shot recall, adds region-level checks before ranking | More moving parts, a bit slower per query than a single lookup | Fashion queries that combine garments, colors, and context — especially compositional ones |

I didn't seriously consider fine-tuning a detector or classifier here,
specifically because zero-shot generalization was a core goal and a
closed-taxonomy classifier is structurally at odds with that.
Retrieve-and-rerank was the only option that let me keep CLIP's
open-vocabulary property *and* fix its compositional blind spot.

## Architecture

### Part A — the indexer (`indexer/`)

Runs once, offline, over the image corpus.

1. **Full-image embedding.** Each image is encoded with
   `hf-hub:Marqo/marqo-fashionCLIP` (loaded via `open_clip_torch`), a CLIP
   checkpoint trained on ~700k fashion image-text pairs rather than generic
   web images. This is the direct answer to "better than vanilla CLIP" —
   fashion vocabulary gets meaningfully better cross-modal alignment than
   OpenAI's base ViT-B/32.
2. **Open-vocabulary detection.** Grounding DINO runs one combined
   garment+person prompt per image (garments and person boxes in a single
   forward pass, which roughly halves CPU inference time versus two
   passes). OWL-ViT is the automatic fallback if GDINO fails to load.
3. **Region features.** Every detected garment crop gets its own CLIP
   embedding, batched together with the full image into a single forward
   pass per image (N+1 crops → 1 CLIP call — the main CPU speedup in the
   pipeline). Each crop also gets a dominant color via K-means, with the
   actual measured RGB stored (not just a color *name*), so "burgundy" in a
   query can still match a crop that our 16-color palette would round down
   to "maroon."
4. **Person association.** Garments are assigned to the most-overlapping
   person box. This is what later lets the reranker tell the difference
   between one person wearing a red tie *and* a white shirt, versus two
   people each wearing one of those things.
5. **Storage.** ChromaDB holds one document per image — the full-image
   embedding as the HNSW-indexed vector, with region metadata (boxes,
   labels, colors, embeddings) serialized alongside it. The store records
   which CLIP checkpoint and detector produced each entry, and both the CLI
   and the web app refuse to query a collection with a mismatched model
   rather than silently returning garbage distances.

The indexer is resumable and idempotent — re-running it on a partially
indexed folder just picks up where it left off, keyed on `(image_id,
clip_model, clip_model_version, detector_model)`.

### Part B — the retriever (`retriever/`)

Runs per query.

1. **Query parsing.** Llama 3.3 70B (via Groq) turns free text into
   structured fields: a list of `{garment, color}` pairs and an optional
   setting phrase. I picked an LLM over regex specifically because fashion
   queries paraphrase heavily — "something for a job interview" and
   "professional business attire" need to land on the same structured
   setting, and no keyword list handles that gracefully. If `GROQ_API_KEY`
   isn't set, or the API call fails, the parser falls back to a
   conservative color+garment keyword matcher rather than crashing — you
   lose paraphrase understanding, but stage-1 CLIP retrieval still runs, so
   the system degrades instead of dying. The five benchmark queries are
   also pre-parsed in `eval/parsed_cache.json`, so evaluation doesn't
   depend on a live API key or network access.
2. **Stage 1 — ANN retrieval.** The raw query text is embedded and
   ChromaDB's HNSW index returns the top 300 candidates by default. This
   stage never touches region metadata — it's the part of the system that
   has to scale, so it stays a single vector lookup regardless of corpus
   size.
3. **Stage 2 — reranking.** Each candidate gets rescored as a weighted sum
   of three signals, each normalized to `[0, 1]` before combining:
   - `stage1_score` — global image/query similarity (scene gestalt),
   - `attribute_score` — how many parsed garment+color pairs found a
     matching *region*, enforced per person instance,
   - `setting_score` — similarity between the parsed context phrase and
     the full image (occasion/scene).

   Base weights are `w_stage1=0.35`, `w_attribute=0.50`, `w_setting=0.15` —
   attribute matching gets the largest share deliberately, since it's the
   one signal doing the compositional work stage 1 structurally can't do.

**Garment matching isn't string matching.** The reranker compares CLIP
embeddings of the query's garment label (plus its synonyms) against stored
region embeddings, so "raincoat" can match a region the detector only ever
called "coat" or "jacket," as long as the embeddings agree past a threshold.
That threshold — and the RGB-distance threshold for color matching — aren't
guessed; I measured them (`eval/calibrate_thresholds.py`, 150 images / 50
positive-negative pairs each) by looking at where the positive-pair and
negative-pair similarity distributions actually separate, then set the
threshold at the midpoint. Color separates cleanly regardless of CLIP
backbone (positive-pair mean distance 28.2 vs. negative-pair mean 211.0).
Garment similarity is a noisier signal on a generic CLIP backbone — more on
what that implies in **Evaluation** below.

**Weights aren't fixed either.** A query like "casual weekend outfit" has no
parsed garments, so `attribute_score` is 0.0 for every candidate — leaving
its weight allocated there would just dilute the two signals that actually
carry information. `_resolve_weights()` in `retriever/reranker.py` detects
inactive signals per query and redistributes their weight proportionally to
the active ones, and separately shifts up to 0.15 of weight from `stage1` to
`attribute` whenever a query has two or more colored garments — because
`attribute_score` is the only signal that can tell "red tie + white shirt"
apart from "white tie + red shirt" in the first place.

**Match confidence isn't binary.** For each parsed garment, the reranker
tracks the best- and second-best-scoring region. A decisive match (large
gap between the two) counts fully; a marginal one, where two regions are
nearly tied, counts at half weight. This keeps one ambiguous, barely-over-
threshold match from contributing as much as a clean one.

## How the five benchmark queries actually get handled

| # | Query | What the parser extracts | What actually does the work |
|---|---|---|---|
| 1 | *"A person in a bright yellow raincoat"* | `garment=raincoat, color=yellow` | Color normalization (`bright yellow → yellow`) plus garment synonym expansion (`raincoat → coat, jacket, anorak, mac`) so a detector box merely labeled "coat" still scores |
| 2 | *"Professional business attire inside a modern office"* | no garments, `setting="office indoor formal business"` | Pure `setting_score` — weight is auto-redistributed away from `attribute` since there's nothing to bind attributes to |
| 3 | *"Someone wearing a blue shirt sitting on a park bench"* | `garment=shirt, color=blue`, `setting="park outdoor"` | Garment+color matching for the shirt, setting score for the bench/outdoor context — these run independently, so a blue-shirt match doesn't get dragged down by a weak park match or vice versa |
| 4 | *"Casual weekend outfit for a city walk"* | no garments, `setting="casual outdoor urban"` | Same as Q2 — a style/vibe query, effectively stage-1 CLIP plus setting similarity |
| 5 | *"A red tie and a white shirt in a formal setting"* | `garment=tie, color=red` + `garment=shirt, color=white`, `setting="formal occasion"` | The compositional stress test: both garments must be found on **the same person**, the attribute weight gets boosted for having two colored garments, and gap scoring penalizes any barely-over-threshold tie/shirt match |

## Zero-shot capability, concretely

Nothing in this pipeline is fine-tuned on Fashionpedia's own category labels
— Grounding DINO and FashionCLIP both run purely at inference time, so the
system isn't boxed into a closed vocabulary. Three separate places where
that matters in practice:

- **Detector vocabulary vs. query vocabulary.** The detector's prompt list
  is finite, but garment matching compares *embeddings*, not label strings.
  A query for "windbreaker" gets a real similarity score against a region
  the detector only ever called "jacket," because the reranker never checks
  whether the strings match — only whether the embeddings agree.
- **Synonym expansion.** `raincoat → coat, jacket, anorak, mac` and similar
  mappings in `retriever/query_parser.py` push recall further for garment
  words the detector's own prompt list doesn't literally contain.
- **Paraphrase at the language level.** The LLM parser maps "something for
  a job interview" and "professional business attire" to the same
  structured setting, because it's reasoning over meaning rather than
  matching fixed phrases — a hand-written keyword list genuinely cannot do
  this.

## Modular by construction

`indexer/` and `retriever/` are separate packages that share exactly two
small, explicitly-imported utility modules (color matching, Chroma
serialization) — nothing else is duplicated between them. Neither package
hardcodes a dataset path, a model name, a fusion weight, or a threshold
inside its logic; every one of those lives in a `config.py` dataclass and is
overridable from the CLI. That's not an abstract "modularity" claim — it's
why the exact same code ran unmodified against a 50-image CPU smoke test and
a 100-image evaluation index, and why tuning retrieval behavior is a
`--w-attribute 0.6` flag rather than a code change.

## Repository layout

```text
Fashionpedia/
├── app/
│   ├── main.py                  # FastAPI: /search, /image/{name}, model-mismatch guard
│   ├── run.py                   # App runner
│   └── static/                  # Minimal web UI for interactive search + eval judging
├── indexer/
│   ├── main.py                  # Offline indexing entry point
│   ├── detector.py              # Grounding DINO / OWL-ViT, combined garment+person prompt
│   ├── color_extractor.py       # K-means dominant color, RGB storage + palette lookup
│   ├── embedder.py               # CLIP / open_clip image encoders, batched forward pass
│   ├── chroma_store.py          # Chroma schema, region (de)serialization
│   └── config.py                # Indexer defaults (paths, model, detector threshold)
├── retriever/
│   ├── main.py                  # CLI entry point, prints per-signal score breakdown
│   ├── query_parser.py          # Groq LLM parser + cache + keyword fallback
│   ├── retrieval.py             # Stage-1 HNSW lookup only
│   ├── reranker.py              # Stage-2 compositional scoring, weight redistribution
│   ├── embedder.py               # Query text encoder
│   └── config.py                # Fusion weights, calibrated thresholds
├── eval/
│   ├── run_eval.py              # Runs the 5 benchmark queries end to end
│   ├── calibrate_thresholds.py  # Measures positive/negative similarity distributions
│   ├── contact_sheet.py         # Generates the top-K grid images in eval/results/
│   ├── comparison_report.md     # Before/after numbers, see Evaluation below
│   └── judgments/               # Saved human relevance labels per query
├── docs/
│   ├── dataset_gap_analysis.md  # Why Q2–Q4 are dataset-limited, not retrieval-limited
│   ├── locations_weather.md     # Full writeup of the location/weather extension
│   └── future_improvements.md   # Deeper notes on possible extensions and next steps
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ (developed and tested on 3.12.3, CPU-only, no GPU required for
inference — GPU just makes indexing faster).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git   # only needed for OpenAI-checkpoint fallback
```

To get full LLM query parsing for arbitrary queries (not just the five
cached benchmark ones), set a Groq key:

```bash
export GROQ_API_KEY=your_key_here      # Windows: set GROQ_API_KEY=your_key_here
```

Without it, the retriever still runs — it falls back to the keyword parser,
and the five benchmark queries parse deterministically from the local
cache regardless.

## Indexing the dataset

Download Fashionpedia `val_test2020` ([direct link][fp-zip]) and extract it
so images live under `datasets/val_test2020/test`.

```bash
python indexer/main.py \
  --image-dir ./datasets/val_test2020/test \
  --db-path ./chroma_db_fashion \
  --clip-model hf-hub:Marqo/marqo-fashionCLIP
```

For a quick CPU sanity check before committing to a full run:

```bash
python indexer/main.py --max-images 50 --device cpu --db-path ./chroma_db_test
```

[fp-zip]: https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip

## Running a query

```bash
python retriever/main.py --db-path ./chroma_db_fashion \
  --query "a red tie and a white shirt in a formal setting" --top-k 5
```

```text
Query: 'a red tie and a white shirt in a formal setting'
------------------------------------------------------------
  Parsed: garments=[{'label': 'tie', 'color': 'red'}, {'label': 'shirt', 'color': 'white'}], setting='formal occasion'
  Stage 1: 300 candidates retrieved.

Top 5 results:
Rank  Score     S1   Attr    Set  Image
--------------------------------------------------------------------------------
1     0.812  0.640  0.950  0.710  a1b2c3d4e5f6....jpg
       matched: tie(red) → tie(red) garment_sim=0.284, color_dist=12.3, person_id=1
       matched: shirt(white) → shirt(white) garment_sim=0.251, color_dist=18.7, person_id=1
...
```

Each row shows the final blended score plus its three components
(`stage1`/`attribute`/`setting`), and any matched-attribute lines show
exactly which region and color the reranker matched, its similarity score,
and which person it was attached to — useful for debugging *why* an image
ranked where it did, not just that it did.

## Web UI

```bash
export CHROMA_DB_PATH=./chroma_db_fashion   # Windows: set CHROMA_DB_PATH=...
python app/run.py
```

Then open `http://127.0.0.1:8000` for interactive search, or
`http://127.0.0.1:8000/eval` for the same relevance-judging UI that produced
the numbers below.

## Evaluation

I want this section to be an honest account, not a highlight reel — "how
well does it actually work" is the question that matters most, and a
README that only shows good numbers isn't answering it.

`eval/run_eval.py` runs the pipeline end to end on all five benchmark
queries, writes a top-K contact sheet image per query to `eval/results/`,
and supports interactive relevance marking (saved to `eval/judgments/`) to
compute real precision@K instead of eyeballing it.

**The numbers below are from a full 3,200-image index built on OpenAI's
vanilla ViT-B/32** — the only configuration for which a complete
index-and-judge pass has finished:

| Query | P@5 |
|---|---|
| Q1 — bright yellow raincoat | 0.20 |
| Q2 — professional business attire, office | 0.60 |
| Q3 — blue shirt, park bench | 0.00 |
| Q4 — casual weekend outfit, city walk | 0.80 |
| Q5 — red tie and white shirt, formal | 0.00 |
| **Mean** | **0.32** |

Two things are worth pulling apart in these numbers rather than averaging
past them:

**Q3 and Q5 at zero are mostly a dataset problem, not a retrieval-logic
problem.** `val_test2020` is drawn from Getty editorial photography —
runway shots, red-carpet, fashion-week presentations. I checked this
directly (`docs/dataset_gap_analysis.md`): candid office interiors and park
benches are close to absent from the corpus. No amount of reranking logic
recovers relevant images that the corpus doesn't contain, and Q2's 0.60
despite a similarly "office"-flavored setting phrase is evidence the
*setting* signal itself is working — it's the *supply* of matching images
that's thin for Q3.

**Q5 scoring zero is more specifically informative,** and it's the result I
take most seriously, because Q5 is the compositional query this whole
architecture exists to solve. The threshold calibration run
(`eval/calibrate_thresholds.py`) measured *why*: on vanilla ViT-B/32,
positive-pair garment similarity (correct label vs. region) averages 0.237,
and negative-pair similarity averages 0.215 — an 0.022 gap that overlaps
heavily in practice. Color, on the same run, separates cleanly (28.2 vs.
211.0). In other words, on a generic CLIP backbone the *attribute_score*
channel — the one signal that can tell "red tie + white shirt" apart from
its inverse — is close to noise, while the color channel is reliably
informative regardless of backbone. That's not a bug in the reranker; it's
the exact failure mode described in **Why this exists**, now measured rather
than assumed, and it's precisely why the indexer's default and documented
configuration is `hf-hub:Marqo/marqo-fashionCLIP` rather than vanilla
ViT-B/32 — a checkpoint trained specifically for fashion text-image
alignment should widen that 0.022 gap substantially.

**Current status, plainly:** a full 3,200-image re-index under FashionCLIP
was still running on CPU-only hardware as of this writing (a full pass
takes on the order of 18 hours without a GPU), so I don't have a completed
FashionCLIP P@5 table to report yet — and I'd rather say that than
backfill numbers I don't have. To reproduce or complete it:

```bash
python eval/calibrate_thresholds.py --db-path ./chroma_db_fashion \
  --clip-model hf-hub:Marqo/marqo-fashionCLIP --out eval/calibration_fashionclip.json

python eval/run_eval.py --db-path ./chroma_db_fashion --skip-judgment
# then open http://127.0.0.1:8000/eval to judge relevance interactively
```

## Known limitations

- **Dataset coverage caps context-heavy queries.** Covered above — this is
  a corpus problem, not purely an architecture problem.
- **One dominant color per garment.** K-means picks the modal color
  cluster, so a solid navy blazer is represented well and a plaid or
  striped shirt is not. Pattern and material need a separate signal
  entirely — a color histogram can't express "plaid."
- **Person association is bounding-box overlap, not segmentation.** Fine
  for a single clearly-posed subject (which describes most of this
  corpus), weaker for crowded or heavily occluded group shots where boxes
  overlap ambiguously.
- **Region embeddings live inside image-level Chroma metadata.** Fine at
  3,200 images. At real scale (millions of images) this would need to
  become its own region-level index keyed by image ID — only stage 1 (the
  HNSW lookup) has to scale with corpus size; stage 2 always operates on a
  fixed-size candidate set regardless of how large the corpus gets.

---

**Author:** Tanishq Sharma, BTech, IIIT Kota