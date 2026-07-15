# Where I Would Take This Given Another Week

This document covers concrete improvements that did not make the initial
implementation, ordered roughly by expected impact per day of effort.

---

## 1. Fine-tune CLIP on the Fashionpedia labeled train split

The current pipeline uses the off-the-shelf OpenAI CLIP checkpoint, which was
trained on 400M internet image-text pairs. Its fashion vocabulary is reasonable
but not specialized: it may conflate "blazer" with "jacket" in embedding space
and has no awareness of Fashionpedia's specific 294-category taxonomy.

The labeled train split (`train2020`) has per-image segmentation masks with
fine-grained attribute labels (not just category names but material, pattern,
reflectance). This is exactly the supervision needed to teach CLIP better
garment discrimination.

**Approach**: Start from the ViT-L/14 checkpoint and fine-tune with a
contrastive objective on (image crop, attribute label) pairs constructed from
the train split's masks. Each masked crop paired with its full attribute label
string (e.g., "silk blazer with notch lapel, striped pattern, solid lining")
forms one positive pair. Negatives are other crops from the same batch.

**What this buys**: After fine-tuning, the garment similarity threshold in the
reranker would need recalibration (the cross-modal cosine distribution shifts),
but matches would be more semantically precise. The color K-means step might
become less necessary if the fine-tuned model learns to associate attribute text
with visual color properties directly.

**Practical note**: The train split has ~45,000 images with masks, which is
enough for a useful fine-tune at a few hundred dollars of compute. Training for
3-5 epochs with a learning rate of 1e-6 on the visual encoder (keeping the text
encoder frozen or lightly tuned) is a reasonable starting point.

---

## 2. Replace the hand-weighted score combination with a learned reranker

The current final score is a fixed weighted sum:

    final = 0.50 * stage1 + 0.35 * attribute + 0.15 * setting

These weights were chosen by reasoning about relative signal quality, not by
fitting to data. They are wrong for at least some query types: for pure vibe
queries ("casual weekend outfit") the attribute weight should be near zero
because there are no parsed attributes, but the formula still applies 0.35 * 0
regardless.

**Approach**: Replace the weighted sum with a small pointwise regression model
(a 2-layer MLP or a gradient-boosted tree) trained on click or relevance data.
Features would be the three signal scores plus auxiliary features: number of
parsed garments, whether LLM parsing succeeded, number of detected regions in
the candidate, etc. Target labels would come from human relevance judgments
collected via the `run_eval.py` judgment mechanism already implemented.

With the five benchmark queries and 10 judgments each, you have 50 labeled
pairs — not enough to train a reranker that generalizes. You would need at least
a few hundred diverse queries with judgments. A realistic roadmap: deploy the
system with the fixed weights, collect judgments passively, retrain the reranker
every week.

**Alternative**: A LambdaRank model (pairwise rather than pointwise) trained on
ranked pairs tends to generalize better from smaller judgment sets. LightGBM has
a native LambdaRank objective that fits in a few lines of code once the feature
vectors are assembled.

---

## 3. Better negative mining for the attribute matching step

The GDINO detection pass finds garments present in the image. The reranker then
checks whether query attributes match detected regions. The problem: an image
with 5 detected garments will almost always match at least one attribute by
chance, because the garment embedding space is not sparse — "shirt" in CLIP
space is close to many casual upper-body garments.

**Approach**: For each query garment embedding, explicitly compute the similarity
against *all* detected regions, not just those that exceed the threshold, and
weight the attribute score by the gap between the best match and the second-best
non-matching region. A large gap indicates a clear match; a small gap indicates
the threshold is being hit by a marginal region.

More aggressively: use hard negative mining at indexing time. For each detected
region, find its nearest neighbors in the CLIP embedding space that are *not*
from the same garment category. Store the top-k hard negatives per region. At
retrieval time, penalize candidates where the query embedding is closer to a hard
negative than to the matched region. This is the same technique used in metric
learning to improve retrieval precision.

---

## 4. Human-in-the-loop relevance feedback

The current pipeline is open-loop: it produces results and the user either
accepts them or reformulates the query. A relevance feedback round can
substantially improve precision at minimal cost.

**Approach**: After returning the initial top-k results, mark some as relevant
(R+) and some as irrelevant (R-). Use the Rocchio update rule to adjust the
query embedding:

    q_new = alpha * q_orig + beta * mean(R+ embeddings) - gamma * mean(R- embeddings)

Re-run stage 1 with q_new as the query vector. Stage 2 reranking runs on the
new candidate set. This is a one-round pseudo-relevance feedback loop with no
model retraining.

The `run_eval.py` judgment infrastructure already collects binary relevance
signals; wiring them into a Rocchio update requires about 20 lines of numpy.

**Known limitation**: Rocchio works well in dense retrieval settings where the
embedding space is roughly Euclidean. CLIP's embedding space is known to have
hub structure (some image embeddings are near-neighbors to many query embeddings
regardless of content), which limits the effectiveness of mean-based feedback.
A more robust alternative is to embed the positive images through the CLIP visual
encoder and use their mean as an additional query term, essentially doing image-
to-image retrieval seeded by the feedback set.

---

## 5. Expand the garment detection vocabulary

The detector prompt currently covers ~35 garment categories. Accessories
completely outside this neighborhood — sunglasses, jewelry, watches, belts with
distinct patterns, hair accessories — will not be detected regardless of how good
the matching logic is.

**Approach**: Run a second detection pass with a dedicated accessory prompt:
`"sunglasses . glasses . earrings . necklace . bracelet . watch . ring . belt .
hairband . headphones"`. Add results to the region list with a category tag
distinguishing them from primary garments, so the reranker can weight them
differently (an accessory match might be worth 0.5x a garment match in the
attribute score).

This does not require any model changes — it is purely a prompt engineering and
metadata schema extension.

---

## Known limitations not worth fixing for this submission

**Multi-color garments**: K-means returns the single dominant cluster. A plaid
shirt or striped garment collapses to whichever color has the most pixels. A
query for "striped shirt" would require texture classification, not color
extraction. There is no cheap fix within the K-means framework; a segmentation-
aware texture classifier is a separate capability.

**Garment vocabulary ceiling**: Even with the extended prompt, any garment type
whose visual appearance GDINO has not seen enough of during pretraining will
produce weak detections. Highly domain-specific categories (traditional garments
from underrepresented cultures, niche activewear) are the most vulnerable.

**Color naming precision**: The synonym lookup table covers ~60 color words. An
unusual fashion term not in the table falls back to CSS4 web colors, which do
not cover fashion-specific terms like "ecru", "champagne", "mauve", or "camel"
completely. When a color word cannot be resolved to RGB, color matching is skipped
for that attribute and the system logs a warning — it degrades gracefully, but
it means a query for "champagne gown" gets no color-matching signal.

**No temporal or seasonal awareness**: The pipeline has no concept of seasons or
trends. "Summer outfits" and "winter outfits" are handled as setting phrases
through the CLIP similarity score, which captures some visual correlation (lighter
colors, fewer layers) but cannot reason about suitability in the way a person can.
