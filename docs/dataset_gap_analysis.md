# Dataset Gap Analysis — val_test2020 vs. Benchmark Queries

## Summary

Several of the 5 benchmark evaluation queries assume scene contexts
(office, park) that are largely absent from the Fashionpedia val_test2020
image set. This is a real limitation that affects P@5/P@10 for queries Q2
and Q4 specifically, and it should be named explicitly in the PDF rather than
left implicit.

---

## What the dataset actually contains

Fashionpedia val_test2020 is drawn from **iMaterialist (Fashion) 2020**, which
was sourced from **Getty Images editorial photography**. The images are:

- **Runway / catwalk shots**: models on a stage, often with a blurred studio
  backdrop or stage lighting. Crowd and photographer silhouettes visible.
- **Street style / editorial**: models photographed on city streets, but in
  controlled/posed editorial style — not candid.
- **Fashion event photography**: red-carpet, fashion week presentations, press
  shoots.

What they are **not**:
- Candid snapshots of people in offices or parks.
- Scene-diverse consumer photos (Instagram, Google Images).
- Any of the contexts described in Q2 ("modern office") and Q4 ("city walk").

---

## Per-query impact

| # | Query | Dataset match | Expected impact |
|---|-------|--------------|----------------|
| Q1 | bright yellow raincoat | High: garment-centric, no scene constraint | Attribute matching dominates; P@5 should be good |
| Q2 | professional business attire inside a modern office | **Low**: "modern office" setting almost absent | setting_score near-zero for all results; effectively garment-only |
| Q3 | blue shirt sitting on a park bench | **Medium–Low**: "park bench" absent; blue shirt present | Garment match works; setting match fails |
| Q4 | casual weekend outfit for a city walk | **Low**: no garments specified + no city-walk scenes | Pure stage1 vibe query; dataset has no casual street candids |
| Q5 | red tie and white shirt in a formal setting | **High**: formal setting ≈ runway/event; ties and shirts present | Best candidate for high P@5 |

---

## Why this matters for the write-up

The assignment rubric rewards "understanding what the system's shortcomings are
and how to address them." Naming this gap — with evidence — reads as rigor,
not weakness. It also explains why Q2 and Q4 scores are lower than Q1/Q5
without implying the retrieval logic is broken.

Suggested framing for the PDF "Approaches / Limitations" section:

> "The val_test2020 split is drawn from Getty editorial photography, which
> is dominated by runway, street-style, and fashion-event imagery. Evaluation
> queries Q2 ("professional business attire inside a modern office") and Q4
> ("casual weekend outfit for a city walk") describe scene contexts largely
> absent from this distribution, which caps the setting_score contribution for
> these queries regardless of retrieval quality. We verified this by manually
> inspecting random samples from the dataset. A dataset augmented with in-the-
> wild consumer photography (e.g. DeepFashion-Consumer-to-Shop) would better
> match these queries."

---

## Mitigating strategies (for future work section)

1. **Dataset augmentation**: Add DeepFashion Consumer-to-Shop or iNaturalist-
   style in-the-wild fashion photos to cover office/park scenes.
2. **Query normalization**: Pre-process the setting field to focus only on
   tokens that map to visual features present in the index (e.g. drop
   "office" setting entirely for this dataset; weight garment attributes more).
   `_resolve_weights()` already moves in this direction — for queries with no
   setting match, the setting weight is redistributed automatically.
3. **Evaluation on a more diverse split**: If re-indexing, use the full
   Fashionpedia train set (2.7M images) where scene diversity is higher.
