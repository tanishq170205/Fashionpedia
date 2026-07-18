# Multimodal Fashion and Context Retrieval

This repository implements a fashion image search system for the Glance ML
internship assignment. The goal is to retrieve images from natural language
queries that combine clothing attributes, colors, and scene context, for
example:

- "A person in a bright yellow raincoat."
- "Professional business attire inside a modern office."
- "Someone wearing a blue shirt sitting on a park bench."
- "A red tie and a white shirt in a formal setting."

The core idea is a two-stage retrieve-and-rerank pipeline. A fashion-tuned CLIP
model gives broad zero-shot recall, while a second stage checks garment-level
matches using detected regions, dominant colors, and person association. This is
meant to address the main weakness of a plain CLIP baseline: CLIP is strong for
overall visual similarity, but it often struggles to bind attributes to the
right garment in compositional queries.

## Why Not Plain CLIP?

A simple baseline would encode each image and the query with CLIP, then rank by
cosine similarity. That is fast and scalable, but it has two issues for this
assignment.

First, CLIP collapses the whole query into one vector. Queries like "red shirt
and blue pants" and "blue shirt and red pants" contain nearly the same words, so
their embeddings can be very close even though the intended outfits are
different.

Second, scene and clothing attributes compete inside the same embedding. For a
query like "professional business attire inside a modern office", the model may
retrieve office-like images even when the clothing is wrong, or formal clothing
even when the setting is wrong. The system here separates those signals so they
can be checked independently.

## Architecture

### Part A: Indexer

The indexer converts raw images into a searchable ChromaDB collection.

1. Full-image embedding: each image is encoded with
   `hf-hub:Marqo/marqo-fashionCLIP`, a fashion-domain CLIP checkpoint loaded
   through `open_clip_torch`.
2. Open-vocabulary detection: Grounding DINO detects garment regions such as
   shirts, jackets, pants, dresses, coats, ties, shoes, and bags. OWL-ViT is
   available as a fallback detector.
3. Region features: each detected garment crop receives its own CLIP embedding.
   The dominant region color is extracted with K-means and stored as RGB.
4. Person association: detected garments are assigned to person boxes by overlap.
   This lets the reranker verify that multiple requested garments belong to the
   same person.
5. Vector storage: ChromaDB stores one document per image. The full-image
   embedding is the ANN vector; region metadata is serialized alongside it.

### Part B: Retriever

The retriever answers a natural language query in two stages.

1. Query parsing: Llama 3.3 70B through Groq parses the query into structured
   fields: garment/color constraints plus an optional setting phrase. The five
   benchmark queries are also cached in `eval/parsed_cache.json` so demos remain
   deterministic without an API call.
2. Stage 1 retrieval: the raw query is embedded with FashionCLIP and ChromaDB's
   HNSW index returns a broad candidate set, defaulting to 300 images.
3. Stage 2 reranking: candidates are rescored with three signals:
   - global image/query similarity from stage 1,
   - garment and color matches over detected regions,
   - setting similarity between the parsed context phrase and the full image.

The reranker expands garment synonyms such as `raincoat -> coat, jacket, anorak`
and normalizes color phrases such as `bright yellow -> yellow` and
`navy blue -> navy`. For garment matching, it compares query label embeddings
against stored region embeddings, so retrieval is not limited to exact detector
labels. Color is checked with Euclidean RGB distance.

For compositional queries, attribute matches are evaluated per person instance.
An image with Person A wearing a red shirt and Person B wearing blue pants should
not receive the same score as one person wearing both requested garments.

## Repository Layout

```text
Fashionpedia/
|-- app/
|   |-- main.py                  # FastAPI search and evaluation API
|   |-- run.py                   # App runner
|   `-- static/                  # Simple web UI
|-- indexer/
|   |-- main.py                  # Offline indexing entry point
|   |-- detector.py              # Grounding DINO / OWL-ViT detection
|   |-- color_extractor.py       # K-means dominant color extraction
|   |-- embedder.py              # CLIP / open_clip image encoders
|   |-- chroma_store.py          # ChromaDB schema and region serialization
|   `-- config.py                # Indexer defaults
|-- retriever/
|   |-- main.py                  # Command-line retrieval entry point
|   |-- query_parser.py          # LLM parser, cache, fallback parser
|   |-- retrieval.py             # Stage 1 ANN lookup
|   |-- reranker.py              # Stage 2 compositional reranking
|   |-- embedder.py              # Query text/image encoder
|   `-- config.py                # Retrieval weights and thresholds
|-- eval/
|   |-- run_eval.py              # Benchmark query runner
|   |-- calibrate_thresholds.py  # Threshold calibration utility
|   |-- contact_sheet.py         # Evaluation contact-sheet generation
|   `-- judgments/               # Saved relevance judgments
|-- docs/
|   |-- dataset_gap_analysis.md
|   |-- locations_weather.md
|   `-- future_improvements.md
|-- generate_pdf.py              # Final report generator
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Set a Groq key if you want full LLM parsing for arbitrary queries:

```bash
# Windows
set GROQ_API_KEY=your_key_here

# macOS/Linux
export GROQ_API_KEY=your_key_here
```

Without `GROQ_API_KEY`, the retriever falls back to a conservative keyword
parser. The five assignment queries still parse deterministically through the
local cache.

## Index the Dataset

Download/extract Fashionpedia `val_test2020` so the images are under
`datasets/val_test2020/test`.

```bash
python indexer/main.py ^
  --image-dir ./datasets/val_test2020/test ^
  --db-path ./chroma_db_fashion ^
  --clip-model hf-hub:Marqo/marqo-fashionCLIP
```

For PowerShell, replace `^` with backticks, or put the command on one line.

The indexer is resumable. It stores model metadata with each document, so a
changed CLIP checkpoint forces re-indexing instead of silently mixing embedding
spaces.

Important: the `chroma_db` folder in this workspace may be a legacy ViT-B/32
index. The final FashionCLIP setup should use `./chroma_db_fashion`, or you
should pass `--clip-model ViT-B/32` when intentionally querying the legacy
index. The app and CLI both guard against model/index mismatches.

For a quick CPU sanity check:

```bash
python indexer/main.py --max-images 50 --device cpu --db-path ./chroma_db_test
```

## Run Retrieval

```bash
python retriever/main.py ^
  --db-path ./chroma_db_fashion ^
  --query "a person in a bright yellow raincoat"

python retriever/main.py ^
  --db-path ./chroma_db_fashion ^
  --query "a red tie and a white shirt in a formal setting" ^
  --top-k 10
```

Each result prints the final score, stage-1 score, attribute score, setting
score, and matched garment regions when available.

## Run the Web UI

```bash
set CHROMA_DB_PATH=./chroma_db_fashion
python app/run.py
```

Then open `http://127.0.0.1:8000`.

## Evaluation

The evaluation runner uses the five prompts from the assignment and can generate
contact sheets for manual judgment.

```bash
python eval/run_eval.py --db-path ./chroma_db_fashion --skip-judgment
```

Saved judgments live in `eval/judgments/`, and generated contact sheets are
written to `eval/results/`.

## What Was Improved Beyond Vanilla CLIP

- FashionCLIP is used instead of a generic CLIP checkpoint.
- Garments are matched at region level rather than only at full-image level.
- Dominant RGB color is stored per garment crop and checked directly.
- Multiple requested garments are matched against the same person instance.
- Query parsing supports zero-shot phrasing through an LLM, with a deterministic
  cache and keyword fallback.
- Synonym expansion improves recall for related garment terms.
- Gap scoring reduces the confidence of ambiguous garment matches where several
  regions score similarly.

## Known Limitations

Fashionpedia `val_test2020` is mostly runway, editorial, and fashion-event
imagery. Office interiors and park-bench scenes are rare, so context-heavy
queries are partly limited by dataset coverage rather than only retrieval logic.

The color extractor stores one dominant color. It does not model multi-color
patterns such as plaid or stripes. Queries about texture or pattern would need a
separate pattern classifier or a stronger region-level vision-language model.

Person association is based on bounding-box overlap. It is useful for ordinary
single-person images, but crowded or heavily occluded scenes would benefit from
pose estimation or segmentation.

At million-image scale, stage 1 remains efficient through HNSW, but storing all
region embeddings inline in Chroma metadata becomes less ideal. A production
version should move regions into a separate region-level index keyed by image ID.

## Future Work

- Add location and weather metadata through EXIF, user tags, historical weather
  joins, or image-based classifiers. Parsed `location` and `weather` fields could
  then be used as Chroma filters before vector retrieval.
- Fine-tune the region encoder on Fashionpedia train masks and attributes to
  improve fine-grained garment discrimination.
- Learn the reranking weights from human relevance judgments instead of keeping
  them hand-set.
- Add pattern and material recognition for prompts like "striped shirt",
  "leather jacket", or "silk dress".
- Add pose-based person/garment association for crowded scenes.
