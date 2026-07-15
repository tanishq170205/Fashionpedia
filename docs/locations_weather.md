# Extending to Real Locations and Weather Conditions

The current pipeline infers scene context from a CLIP text embedding of a parsed
setting phrase like "park" or "office". That works well enough for broad
indoor/outdoor distinctions, but it cannot distinguish a park in Berlin from a
park in Tokyo, a rainy afternoon from a sunny one, or a winter coat worn outdoors
at 5°C from the same coat worn in a cold office. This document describes a
concrete approach to adding real location and weather awareness.

---

## Why the current approach falls short

CLIP's setting score is a similarity between a short phrase and a full scene
embedding. The phrase "park outdoor" is useful but carries no geographic or
meteorological signal — CLIP was not trained to distinguish weather conditions or
specific locations. Two images of someone in a raincoat, one taken in Seattle
rain and one in bright Tokyo sunshine, will score identically on a "rainy day
outdoors" query. The model's scene associations are distributional: "park"
activates greenery and benches, not latitude or season.

---

## What data would actually be needed

**For location:**

1. **GPS EXIF coordinates** — Fashionpedia val_test2020 images do not have EXIF
   GPS data. The images were sourced from social media platforms that strip EXIF
   on upload. Real-world deployment would require user-supplied location tags or
   images from a source that retains EXIF.

2. **A geo-scene classifier** — Given GPS coordinates, a classifier trained on
   geo-tagged image datasets (Im2GPS, GeoEstimation, or the YFCC100M subset with
   GPS metadata) can infer country and region from visual features alone, without
   relying on EXIF. GeoEstimation (Muller-Budack et al., 2018) achieves city-
   level accuracy on typical street photos and is available as a pretrained model.
   For fashion specifically, location markers like signage, architecture, and
   street furniture are more useful than the clothing itself.

3. **Index-time enrichment** — Once a location estimate is available per image,
   store it in the ChromaDB metadata as a `geo_tag` field (country, city, or a
   discrete region enum). At retrieval time the query parser would extract a
   location entity ("Tokyo", "Italy") and filter on the `geo_tag` field before
   or alongside the HNSW lookup.

**For weather:**

1. **Historical weather API join** — If timestamps and GPS are available, the
   OpenWeather historical API (or equivalent) can return temperature, precipitation,
   and cloud cover for any location+time pair. This gets joined to each image
   as structured metadata at index time: `{"weather": "rainy", "temp_c": 12}`.

2. **Weather classification from the image itself** — A classifier trained on
   weather-labeled datasets (DAWN, ACDC, or RESIDE for outdoor conditions) can
   predict weather class (clear, overcast, rain, snow, fog) from pixel statistics.
   The advantage over the API join is that it works without any timestamp or GPS
   data. The disadvantage is that indoor images are ambiguous — a studio photo
   always looks "clear" regardless of outside conditions.

3. **Attribute filtering at retrieval** — A parsed query like "raincoat in wet
   weather" would produce `{"weather": "rainy"}` from the LLM parser. Stage 2
   reranking would check the candidate's stored weather label, adding a binary
   match bonus to the attribute score. This is a metadata filter, not an
   embedding comparison — it is fast and exact, unlike the current setting score.

---

## Concrete implementation path (one week of work)

1. Run GeoEstimation on all 3,200 images to predict country/region; store result
   in ChromaDB metadata as a `region` string field.

2. Add a weather classifier fine-tuned on DAWN/ACDC to the indexer pipeline;
   store predicted weather class in metadata.

3. Extend the query parser prompt to extract optional `location` and `weather`
   fields in its JSON output.

4. Add a pre-filter step in `retrieval.py` before the HNSW query: if a `region`
   or `weather` field is present in the parsed query, pass a Chroma `where` filter
   to restrict the ANN search to documents matching those metadata values. Chroma
   supports scalar metadata filters natively, so this adds no new infrastructure.

5. Update the reranker's attribute score to include location and weather match
   as additional binary terms.

---

## Known failure modes of this approach

- A geo classifier trained on street photography will underperform on studio
  shots, which make up a large fraction of Fashionpedia. Studio images have no
  geographic visual features.
- Weather conditions within indoor images are fundamentally unobservable from
  pixels. If the user queries "what would someone wear at a snowy outdoor event",
  the weather signal in an indoor image is zero regardless of classifier quality.
- EXIF-stripped images from social media (the majority of fashion datasets) have
  no timestamp or GPS, so the weather API join path is unavailable without a
  secondary crowdsourced metadata source (e.g., geolocation via place tagging).
