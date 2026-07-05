"""Cosine similarity search over well embeddings. Uses FAISS if it's
importable, otherwise falls back to a plain numpy dot product + argsort
(fine at this dataset size — a few thousand wells)."""
from collections import Counter

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class SimilarityIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.index = None

    def search(self, query_idx: int, k: int = 10):
        """Returns (neighbor_indices, similarity_scores), self excluded, length <= k."""
        if HAS_FAISS:
            query = self.embeddings[query_idx:query_idx + 1]
            scores, idxs = self.index.search(query, k + 1)
            idxs, scores = idxs[0], scores[0]
        else:
            sims = self.embeddings @ self.embeddings[query_idx]
            idxs = np.argsort(-sims)[:k + 1]
            scores = sims[idxs]

        keep = idxs != query_idx
        return idxs[keep][:k], scores[keep][:k]


def generate_insight(entity_label: str, query_moa, neighbor_moas, similarities) -> str:
    """Deterministic, template-based biological insight paragraph.

    entity_label: "well" or "compound" (for phrasing only).
    query_moa: the query's annotated MoA, or None/empty if unannotated.
    neighbor_moas: list of each neighbor's MoA (None/empty allowed), same
        length and order as `similarities`.
    similarities: list/array of cosine similarity scores for those neighbors.

    No LLM, no external calls — purely arithmetic on the neighbor result,
    phrased cautiously (evidence/hypothesis language, not conclusions).
    """
    if not neighbor_moas:
        return "No neighbors were found to compare against."

    k = len(neighbor_moas)
    avg_sim = float(np.mean(similarities))
    norm_moas = [m if m else "unannotated" for m in neighbor_moas]
    dominant_moa, dominant_count = Counter(norm_moas).most_common(1)[0]
    dominant_frac = dominant_count / k
    has_query_moa = bool(query_moa)
    shared = sum(1 for m in norm_moas if has_query_moa and m == query_moa)

    if dominant_moa == "unannotated" and dominant_frac >= 0.5:
        if has_query_moa:
            return (
                f"This {entity_label} is annotated as {query_moa}. Most of the top {k} nearest "
                f"neighbors ({dominant_count}/{k}) have no annotated mechanism of action, which "
                "limits how much can be concluded here."
            )
        return (
            f"This {entity_label} has no annotated mechanism of action, and neither do most of "
            f"its nearest neighbors. Mean cosine similarity among the top {k} is {avg_sim:.2f}."
        )

    if has_query_moa and dominant_moa == query_moa:
        return (
            f"This {entity_label} lies in a neighborhood enriched for {query_moa}. "
            f"{shared} of {k} nearest neighbors share this annotated mechanism of action and "
            f"appear similar in phenotype, with mean cosine similarity {avg_sim:.2f}. Together "
            f"these neighbors share phenotypic characteristics consistent with a common signature "
            f"for {query_moa} — evidence worth further investigation, not a confirmed mechanism."
        )

    if has_query_moa:
        return (
            f"This {entity_label} is annotated as {query_moa}, but its nearest neighbors appear "
            f"more phenotypically similar to compounds annotated as {dominant_moa} "
            f"({dominant_count} of {k}, mean similarity {avg_sim:.2f}). This may indicate "
            "overlapping or misannotated mechanisms — worth treating as a hypothesis to "
            "investigate, not a conclusion."
        )

    return (
        f"This {entity_label} has no annotated mechanism of action. Its nearest neighbors appear "
        f"similar in phenotype to compounds annotated as {dominant_moa} ({dominant_count} of {k}, "
        f"mean similarity {avg_sim:.2f}), which may indicate a possible hypothesis for its "
        "mechanism worth following up on."
    )
