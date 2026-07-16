"""
Stage 2 reranker.

Takes the stage-1 candidate list and reorders it using structured attributes
extracted from the parsed query. The final score is a weighted combination of
three signals:

  stage1_score:    How similar is the full-image CLIP embedding to the query
                   text embedding? Captures scene gestalt.

  attribute_score: How many of the parsed garment+color pairs have a matching
                   region in the image? Uses semantic CLIP similarity for
                   garment matching (not keyword lookup) and Euclidean RGB
                   distance for color matching. Enforced per person instance.

  setting_score:   How similar is the parsed setting phrase to the full-image
                   CLIP embedding? Captures occasion/context.

All three components are in [0, 1] before combination. This normalization is
what makes the weighted sum meaningful — without it, a raw Chroma L2 distance
(unbounded) mixed with attribute_score (0-1) produces a score that can rank
inverted without any obvious error.

Key design: weight redistribution via _resolve_weights().

Fixed weights waste budget on signals that contribute nothing:
  - "Casual weekend outfit" has no parsed garments → attribute_score == 0.0
    for every candidate; its weight just dilutes stage1 + setting.
  - Queries without a setting → setting_score == 0.0 for every candidate.
_resolve_weights() detects inactive signals and redistributes their weight
proportionally to active ones. For compositional queries (2+ colored garments)
it also shifts weight from stage1 toward attribute because attribute_score is
the ONLY signal that encodes multi-garment composition (stage1 cannot tell
"red tie + white shirt" from "white tie + red shirt" apart).

Key design: garment matching uses stored region_embedding vectors, not string
comparison. This means "windbreaker" in a query can match a detected "jacket"
region if CLIP agrees they are visually similar — the detector's vocabulary
ceiling does not become a retrieval ceiling. The threshold (0.20 cosine) is
the only bottleneck, and it can be tuned.

Key design: attribute matching is per person instance. Without this, an image
with Person A in a red shirt and Person B in blue pants would score as a full
match for "red shirt and blue pants". The person_id field on each region lets
us enforce that all matched attributes come from the same individual.
"""

from __future__ import annotations

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# sys.path adjustment so retriever modules can be imported regardless of CWD.
sys.path.insert(0, os.path.dirname(__file__))

from query_parser import ParsedQuery
from embedder import encode_text
import chromadb

# Import color utilities from the indexer. Since both packages share the
# same repo root, we add the indexer directory to the path here.
_INDEXER_DIR = os.path.join(os.path.dirname(__file__), "..", "indexer")
sys.path.insert(0, _INDEXER_DIR)
from color_extractor import color_name_to_rgb, rgb_distance
from chroma_store import load_regions


@dataclass
class RankedResult:
    image_path: str
    image_id: str
    final_score: float
    stage1_score: float
    attribute_score: float
    setting_score: float
    # Effective weights used for this specific query (after redistribution).
    w_stage1_used: float = 0.0
    w_attribute_used: float = 0.0
    w_setting_used: float = 0.0
    # Human-readable breakdown of which attributes matched and on which person.
    matched_attributes: list[dict] = field(default_factory=list)


def _resolve_weights(
    base_w_stage1: float,
    base_w_attribute: float,
    base_w_setting: float,
    has_garments: bool,
    has_setting: bool,
    n_colored_garments: int,
) -> tuple[float, float, float]:
    """
    Dynamically redistribute fusion weights based on which signals are active.

    Static weights waste budget on signals that contribute nothing:
    - A vibe-only query ("casual weekend") has no garments → attribute_score
      is always 0.0 for every candidate; its weight dilutes the other signals.
    - A query with no setting → setting_score is always 0.0.

    Redistribution rules (applied in order):
    1. Compositional boost: if query has 2+ colored garments, shift up to 0.15
       from stage1 to attribute. attribute_score is the ONLY signal that can
       tell "red tie + white shirt" from "white tie + red shirt" apart. stage1
       cannot. Giving it more weight on compositional queries directly raises P@5
       for queries like Q5 in the benchmark.
    2. Dead weight: if no garments parsed, redistribute w_attribute → stage1 +
       setting proportionally. If no setting parsed, redistribute w_setting →
       stage1 + attribute proportionally.
    3. Normalize to exactly 1.0 (guards against floating-point drift).

    Returns (w_stage1, w_attribute, w_setting), each in [0, 1], summing to 1.0.
    """
    ws, wa, wsc = base_w_stage1, base_w_attribute, base_w_setting

    # Step 1 — compositional boost.
    # Never take more than 40% of ws to avoid degenerate cases (e.g. ws=0.05).
    if n_colored_garments >= 2:
        boost = min(0.15, ws * 0.40)
        ws -= boost
        wa += boost

    # Step 2a — kill attribute weight if no garments in query.
    if not has_garments:
        total_active = ws + wsc
        if total_active > 1e-9:
            ws  += wa * (ws  / total_active)
            wsc += wa * (wsc / total_active)
        else:
            ws += wa  # edge case
        wa = 0.0

    # Step 2b — kill setting weight if no setting in query.
    if not has_setting:
        total_active = ws + wa
        if total_active > 1e-9:
            ws += wsc * (ws / total_active)
            wa += wsc * (wa / total_active)
        else:
            ws += wsc
        wsc = 0.0

    # Step 3 — normalize to 1.0.
    total = ws + wa + wsc
    if total > 1e-9:
        ws, wa, wsc = ws / total, wa / total, wsc / total

    return round(ws, 4), round(wa, 4), round(wsc, 4)


def rerank(
    candidates: list[dict],
    parsed_query: ParsedQuery,
    query_embedding: np.ndarray,
    w_stage1: float = 0.35,
    w_attribute: float = 0.50,
    w_setting: float = 0.15,
    color_distance_threshold: float = 80.0,
    garment_similarity_threshold: float = 0.20,
) -> list[RankedResult]:
    """
    Rerank stage-1 candidates using structured attribute matching.

    Args:
        candidates:    Output of retrieval.stage1_retrieve().
        parsed_query:  Output of query_parser.parse_query().
        query_embedding: CLIP text embedding of the raw query (used for setting score).
        w_*:           Base score fusion weights. These are redistributed per-query
                       by _resolve_weights() before scoring — see that function.
        color_distance_threshold: Max Euclidean RGB distance to count as color match.
        garment_similarity_threshold: Min CLIP cosine similarity to count as garment match.

    Returns:
        List of RankedResult, sorted by final_score descending.
    """
    has_garments = len(parsed_query.garments) > 0
    has_setting  = parsed_query.setting is not None
    n_colored    = sum(1 for g in parsed_query.garments if g.get("color"))

    # Resolve effective weights for this query once, before the candidate loop.
    eff_ws, eff_wa, eff_wsc = _resolve_weights(
        base_w_stage1=w_stage1,
        base_w_attribute=w_attribute,
        base_w_setting=w_setting,
        has_garments=has_garments,
        has_setting=has_setting,
        n_colored_garments=n_colored,
    )

    # Pre-compute text embeddings for each parsed garment label.
    # This is done once before the per-candidate loop — not inside it — to
    # avoid redundant CLIP calls (each call is a GPU/CPU forward pass).
    garment_label_vecs: list[np.ndarray] = []
    for g in parsed_query.garments:
        vec = encode_text(g["label"])
        garment_label_vecs.append(vec)

    # Pre-compute setting embedding if a setting was parsed.
    setting_vec: Optional[np.ndarray] = None
    if parsed_query.setting:
        setting_vec = encode_text(parsed_query.setting)

    # Pre-compute reference RGB for each parsed garment's color.
    garment_color_refs: list[Optional[tuple]] = []
    for g in parsed_query.garments:
        if g["color"]:
            ref = color_name_to_rgb(g["color"])
            if ref is None:
                import warnings
                warnings.warn(
                    f"Could not map color word '{g['color']}' to RGB. "
                    "Color matching will be skipped for this attribute.",
                    stacklevel=2,
                )
            garment_color_refs.append(ref)
        else:
            garment_color_refs.append(None)

    results = []

    for candidate in candidates:
        meta = candidate["metadata"]
        distance = candidate["distance"]
        full_image_vec = np.array(candidate["embedding"], dtype=np.float32)

        # stage1_score: convert cosine distance to similarity.
        # With hnsw:space=cosine, distance ∈ [0, 2] for normalized vectors.
        # Clip to [0, 1] so it stays comparable with attribute_score.
        stage1_score = float(np.clip(1.0 - distance, 0.0, 1.0))

        # Deserialize stored regions.
        regions = load_regions(meta)

        # Attribute score — per person instance.
        attribute_score, matched_attrs = _compute_attribute_score(
            regions=regions,
            parsed_garments=parsed_query.garments,
            garment_label_vecs=garment_label_vecs,
            garment_color_refs=garment_color_refs,
            garment_similarity_threshold=garment_similarity_threshold,
            color_distance_threshold=color_distance_threshold,
        )

        # Setting score.
        if setting_vec is not None and full_image_vec is not None:
            # Both vectors are L2-normalized; dot product == cosine similarity.
            raw_cos = float(np.dot(setting_vec, full_image_vec))
            # CLIP cross-modal cosines typically land in [-0.3, 0.4].
            # Rescale from [-1, 1] to [0, 1] so it can combine cleanly.
            setting_score = float(np.clip((raw_cos + 1.0) / 2.0, 0.0, 1.0))
        else:
            setting_score = 0.0

        final_score = (
            eff_ws  * stage1_score
            + eff_wa  * attribute_score
            + eff_wsc * setting_score
        )

        results.append(RankedResult(
            image_path=meta.get("image_path", ""),
            image_id=candidate["id"],
            final_score=round(final_score, 4),
            stage1_score=round(stage1_score, 4),
            attribute_score=round(attribute_score, 4),
            setting_score=round(setting_score, 4),
            w_stage1_used=eff_ws,
            w_attribute_used=eff_wa,
            w_setting_used=eff_wsc,
            matched_attributes=matched_attrs,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


def _compute_attribute_score(
    regions: list[dict],
    parsed_garments: list[dict],
    garment_label_vecs: list[np.ndarray],
    garment_color_refs: list[Optional[tuple]],
    garment_similarity_threshold: float,
    color_distance_threshold: float,
) -> tuple[float, list[dict]]:
    """
    Compute the attribute match score for one candidate image.

    Returns (score, matched_attribute_list).

    The score is computed per person instance: we try to match all parsed
    garment+color pairs against regions belonging to a single person_id.
    The best single-person match score is used (not the sum across persons).

    If there are no parsed garments, attribute_score = 0.0 (the query has no
    structured garment attributes, so this component doesn't contribute).

    If there are no detected regions, attribute_score = 0.0 by the same logic.
    """
    if not parsed_garments or not regions:
        return 0.0, []

    # Group regions by person_id.
    person_groups: dict[int, list[dict]] = {}
    for region in regions:
        pid = region.get("person_id", -1)
        person_groups.setdefault(pid, []).append(region)

    best_score = 0.0
    best_matched = []

    for pid, pid_regions in person_groups.items():
        score, matched = _match_garments_to_regions(
            parsed_garments=parsed_garments,
            garment_label_vecs=garment_label_vecs,
            garment_color_refs=garment_color_refs,
            regions=pid_regions,
            garment_similarity_threshold=garment_similarity_threshold,
            color_distance_threshold=color_distance_threshold,
            person_id=pid,
        )
        if score > best_score:
            best_score = score
            best_matched = matched

    # Fallback for person_id=-1 regions (accessories, no person association):
    # they contribute to a secondary score at 50% weight, so they can break
    # ties without dominating the ranking. Only used if no person-associated
    # match was found.
    if best_score == 0.0 and -1 in person_groups:
        fallback_score, fallback_matched = _match_garments_to_regions(
            parsed_garments=parsed_garments,
            garment_label_vecs=garment_label_vecs,
            garment_color_refs=garment_color_refs,
            regions=person_groups[-1],
            garment_similarity_threshold=garment_similarity_threshold,
            color_distance_threshold=color_distance_threshold,
            person_id=-1,
        )
        best_score = fallback_score * 0.5
        best_matched = fallback_matched

    return best_score, best_matched


def _match_garments_to_regions(
    parsed_garments: list[dict],
    garment_label_vecs: list[np.ndarray],
    garment_color_refs: list[Optional[tuple]],
    regions: list[dict],
    garment_similarity_threshold: float,
    color_distance_threshold: float,
    person_id: int,
) -> tuple[float, list[dict]]:
    """
    For a given set of regions (from one person instance), compute how many
    parsed garment+color pairs find a matching region.

    Garment match: CLIP cosine similarity between query label text embedding
      and stored region_embedding (visual CLIP embedding of the crop) >= threshold.
      This is the core fix for vocabulary-limited matching: "windbreaker" in the
      query can match a "jacket" detection because their CLIP representations are
      nearby in embedding space.

    Color match: Euclidean RGB distance between query color → reference RGB
      and stored color_rgb <= threshold. Using raw RGB distances instead of
      comparing color name strings avoids the "burgundy" vs "maroon" mismatch.
    """
    matched = []
    match_count = 0

    for i, (parsed_g, label_vec, color_ref) in enumerate(
        zip(parsed_garments, garment_label_vecs, garment_color_refs)
    ):
        best_region_for_this_garment = None
        best_garment_sim = 0.0

        for region in regions:
            region_emb = region.get("region_embedding")
            if region_emb is None:
                continue

            # Cosine similarity — both vectors should be L2-normalized from indexing.
            # Re-normalizing here is cheap and guards against any storage rounding.
            rv = region_emb / (np.linalg.norm(region_emb) + 1e-10)
            lv = label_vec / (np.linalg.norm(label_vec) + 1e-10)
            garment_sim = float(np.dot(lv, rv))

            if garment_sim >= garment_similarity_threshold and garment_sim > best_garment_sim:
                best_garment_sim = garment_sim
                best_region_for_this_garment = region

        if best_region_for_this_garment is None:
            continue  # no region passed the garment similarity threshold

        # If a color was specified, check it against the region's measured RGB.
        color_matched = True
        color_distance = None
        if color_ref is not None:
            stored_rgb = best_region_for_this_garment.get("color_rgb")
            if stored_rgb is not None:
                color_distance = rgb_distance(stored_rgb, color_ref)
                color_matched = color_distance <= color_distance_threshold
            else:
                # No color data stored for this region; skip color check.
                color_matched = True

        if color_matched:
            match_count += 1
            matched.append({
                "query_label": parsed_g["label"],
                "query_color": parsed_g["color"],
                "matched_region_label": best_region_for_this_garment["label"],
                "matched_region_color": best_region_for_this_garment.get("color_name"),
                "garment_similarity": round(best_garment_sim, 3),
                "color_distance": round(color_distance, 1) if color_distance is not None else None,
                "person_id": person_id,
            })

    score = match_count / len(parsed_garments)
    return score, matched
