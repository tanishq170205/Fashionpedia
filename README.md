# Fashion Image Search

A two-stage retrieval system for the Fashionpedia dataset. Given a natural
language query, it finds the most relevant images from a corpus of 3,200
fashion photographs using a combination of open-vocabulary garment detection,
per-region CLIP embeddings, and LLM-structured query parsing.

---

## The problem with plain CLIP retrieval

The obvious approach is to encode the query string with CLIP's text encoder,
encode every image with the vision encoder, and return images sorted by cosine
similarity. This works reasonably well for simple queries like "red dress", but
it breaks in at least two specific ways that matter for fashion search.

**Compositional queries collapse into one vector.** CLIP's text encoder produces
a single embedding for the entire input string by averaging token representations.
"Red shirt and blue pants" and "blue shirt and red pants" land in nearly the same
place in embedding space because the tokens contribute additively — the model
sees the same words, just in different order, and the positional structure that
distinguishes them is largely washed out after the transformer's pooling step.
An image of someone in a blue shirt and red pants would score nearly identically
for both queries. The naive baseline cannot tell them apart.

**Scene queries and attribute queries compete for the same vector.** A query like
"professional business attire inside a modern office" mixes a scene context
("modern office") with implicit garment attributes (suit, tie, formal shoes).
CLIP's joint embedding space has no way to weight these two types of signal
independently. In practice, scene-level features tend to dominate because they
are spatially broader and more distinctive in the training data — so this query
often retrieves images that look like offices regardless of what the people in
them are wearing. "Casual weekend outfit for a city walk" is the extreme version
of this: there are no explicit garment attributes at all, only a vague social
context, and plain CLIP has almost no signal to work with.

---

## How this system works

**Indexing stage.** For each image, a Grounding DINO detector (open-vocabulary)
finds individual garment regions: shirt, jacket, pants, dress, etc. For each
detected region, two things are computed: the dominant color of the crop (via
K-means on the pixel values, mapped to an approximate RGB measurement), and a
CLIP visual embedding of just that crop. A separate detection pass finds person
bounding boxes, and each garment region is assigned to the nearest overlapping
person. This person association is what later lets the reranker enforce that
"red shirt and blue pants" means the same person wearing both, not two different
people. The full-image CLIP embedding is also computed to capture scene and
context. Everything is stored in ChromaDB: one document per image, with the
full-image embedding as the primary vector and all garment regions plus their
embeddings serialized into the metadata.

**Retrieval stage.** A query arrives as a raw string. It first goes through a
Groq LLM call (Llama 3.3 70B) that extracts a structured representation: a list
of garment+color pairs ({"label": "shirt", "color": "blue"}) and an optional
setting phrase ("park outdoor", "formal occasion"). This parsing step is the
reason the system can handle "something professional for a job interview" as
well as "navy blazer and gray trousers" — the LLM generalizes to query phrasings
it has not seen before rather than relying on keyword patterns.

The parsed query text is also encoded with CLIP, and the resulting vector is
used for a fast ANN lookup (HNSW index, native to ChromaDB) to retrieve the top
100 candidate images. This first pass is the only step that touches the full
corpus; everything else operates on 100 images.

The top 100 candidates are then reranked using the structured attributes. For
each parsed garment label, its CLIP text embedding is compared against the stored
CLIP visual embedding of every detected region in the candidate image. If the
cosine similarity exceeds a threshold (0.25), it counts as a garment match —
which means "windbreaker" in the query can match a detected "jacket" region if
CLIP thinks they are visually similar, regardless of whether "windbreaker" was
ever in the detector's vocabulary. Color matching works by mapping the query's
color word to an approximate RGB value and comparing it against the stored RGB
measurement of the matched region using Euclidean distance — so "burgundy" and
"maroon" describe the same pixel measurement and match correctly.

The attribute matches are counted per person instance. If the best person-level
attribute match comes from Person 0, who has a red shirt and blue pants, that
full match is credited. If Person 0 has a red shirt and Person 1 (a different
person in the same image) has blue pants, the query "red shirt and blue pants"
gets only a partial match, because no single person wore both.

The final score is a weighted sum of the stage-1 CLIP similarity (0.50), the
attribute match score (0.35), and a setting/scene similarity computed between
the parsed context phrase and the full-image embedding (0.15). All three
components are normalized to [0, 1] before combination, so the weighted sum
is meaningful.

---

## Repository structure

```
Fashionpedia/
├── datasets/
│   └── val_test2020/test/     ← 3200 raw JPGs
├── indexer/
│   ├── main.py                ← Entry point: run this to index
│   ├── detector.py            ← Grounding DINO / OWL-ViT
│   ├── color_extractor.py     ← K-means dominant color
│   ├── embedder.py            ← CLIP image/text encoder
│   ├── chroma_store.py        ← ChromaDB schema and upsert
│   └── config.py              ← All configurable parameters
├── retriever/
│   ├── main.py                ← Entry point: run this to query
│   ├── query_parser.py        ← LLM structured extraction
│   ├── retrieval.py           ← Stage 1 ANN lookup
│   ├── reranker.py            ← Stage 2 attribute reranking
│   ├── embedder.py            ← CLIP text encoder
│   └── config.py
├── eval/
│   ├── run_eval.py            ← Benchmark queries + P@K
│   └── contact_sheet.py       ← Grid image output
├── docs/
│   ├── locations_weather.md   ← How to add geo/weather signals
│   └── future_improvements.md ← One-week improvement roadmap
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install CLIP (not on PyPI)
pip install git+https://github.com/openai/CLIP.git

# 4. Set your Groq API key (required for LLM query parsing)
#    Get a free key at https://console.groq.com
set GROQ_API_KEY=your_key_here   # Windows
# export GROQ_API_KEY=your_key_here   # Linux/Mac
```

The Grounding DINO checkpoint (~340MB) downloads automatically on first use via
the `transformers` library. If it fails or your CUDA build is incompatible,
the detector falls back to OWL-ViT automatically.

---

## Indexing

```bash
cd indexer

# Full run (2-4 hours on GPU, 8-12 hours on CPU)
python main.py --image-dir ../datasets/val_test2020/test --db-path ../chroma_db

# Quick sanity check on 50 images
python main.py --max-images 50 --device cpu

# Point at a different dataset
python main.py --image-dir /path/to/other/images --db-path /path/to/db

# See all options
python main.py --help
```

The indexer is resumable. If it stops partway through, re-running the same
command skips images that are already in the database. If you change the CLIP
model between runs, old entries are overwritten with the new embeddings rather
than silently coexisting (model name + version is part of the skip key).

A `chroma_db/index_run_summary.json` file is written at completion with counts
and timing.

---

## Querying

```bash
cd retriever

python main.py --query "a person in a bright yellow raincoat"
python main.py --query "red tie and white shirt" --top-k 10
python main.py --query "casual weekend look" --no-llm   # keyword fallback only
```

Output includes a per-image score breakdown showing stage-1 similarity,
attribute match score, setting score, and which specific attributes matched
(query label, matched region label, garment similarity, color distance,
person ID).

---

## Evaluation

```bash
cd eval

# Run all 5 benchmark queries, save contact sheets, collect relevance judgments
python run_eval.py --db-path ../chroma_db

# Generate contact sheets only (no judgment prompts)
python run_eval.py --db-path ../chroma_db --skip-judgment

# Use keyword fallback instead of Groq (no API key needed)
python run_eval.py --db-path ../chroma_db --no-llm
```

Contact sheet images are saved to `eval/results/`. Relevance judgments are
saved to `eval/judgments/` and reused on subsequent runs. Precision@5 and
Precision@10 are printed per query and averaged at the end.

---

## Things that did not work on the first attempt

**Color K-means getting thrown off by background pixels.** The initial version
ran K-means on the full bounding box crop. When the detector box was loose —
which happens often for long garments like coats that the detector clips at the
bottom of the frame — the dominant cluster was often the background color rather
than the garment. The fix was to center-crop to 80% of the bounding box before
K-means, which discards the most contaminated border pixels. It is a heuristic
and still fails for very loose boxes, but it improved color accuracy substantially
on the test images.

**Detector missing small accessories.** GDINO at the default threshold of 0.30
rarely detects items smaller than roughly 50×50 pixels in a 640-pixel image,
which means belts, hats in the background, and most bags escape detection unless
they occupy a significant fraction of the frame. Lowering the threshold below
0.25 increased recall but also produced many false positive boxes on background
textures, which added noise to the reranker. The 0.30 default is a compromise;
if accessory retrieval matters, a second detection pass at 0.20 with a restricted
accessory-focused prompt would help without flooding the main garment list.

**Person-garment association failing for accessories.** Items that naturally
extend beyond the body outline — large bags, wide-brim hats, scarves — often
have less than 25% overlap with the person bounding box, which means they get
assigned `person_id = -1`. They still contribute to the attribute score through
the fallback path (at 50% weight), but they cannot contribute to a
multi-attribute person-level match. This is a known limitation of the current
association logic; a segmentation-based approach would handle it correctly.

---

## Where this breaks down at scale

**The metadata retrieval bottleneck.** Stage 2 reranking deserializes the full
region metadata for each of the top-100 candidates, which includes decoding
base64-encoded float32 embeddings. At 3,200 images with an average of 4 regions
each, the base64 decode is fast. At 1M images, stage 1 still returns 100
candidates (HNSW is logarithmic), but those 100 candidates may each have 10-20
regions, and the base64 decode + numpy operations add up. Storing region
embeddings in a separate vector store (a second Chroma collection keyed by
image_id) with a join step at rerank time would be cleaner at scale. The current
inline metadata approach was chosen to keep the schema simple for a 3k-image
corpus.

**HNSW build time and memory.** Chroma's HNSW index is built incrementally as
documents are upserted. At 1M documents with 1024-dimensional ViT-L/14
embeddings, the index occupies roughly 4-8GB of RAM at query time. This is
within range for a single machine with a good server config, but it requires
tuning HNSW's `M` and `ef_construction` parameters (accessible via
`chromadb.Settings`) for the recall/latency tradeoff you want.

**Single-person assumption.** The person-garment association logic assigns each
garment to exactly one person instance. Heavily occluded or crowded scenes
(multiple people overlapping) produce incorrect associations and therefore
incorrect compositionality enforcement. The fix — using a dedicated pose
estimator to get per-person keypoints and assign garments by keypoint proximity
— is accurate but expensive.

**Multi-color garments are unsupported by design.** K-means extracts one
dominant color. A plaid shirt stores one color (probably the background color
of the plaid). A striped top stores whichever stripe color has more total pixels.
There is no way to query for "striped" or "plaid" through the current color
pipeline. Handling this would require a texture classifier, not a color extractor.

**The LLM parse adds ~0.5s per query.** For interactive use this is acceptable.
For high-throughput batch retrieval (hundreds of queries per second), the Groq
call is the bottleneck. Options: cache parsed results for repeated queries, run
a local 7B model instead, or pre-parse a fixed query library.

---

## License and attribution

Built on top of:
- [OpenAI CLIP](https://github.com/openai/CLIP) (MIT)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Apache 2.0)
- [ChromaDB](https://github.com/chroma-core/chroma) (Apache 2.0)
- [Fashionpedia](https://fashionpedia.github.io/) dataset (CC BY 4.0)
