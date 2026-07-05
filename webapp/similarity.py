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


def _cap_token(t):
    if not t:
        return t
    if any(c.isupper() for c in t):
        return t  # already-cased acronym (HDAC, CDK, MAPK, ...) — leave as-is
    if len(t) > 1 and t[0].isalpha() and t[1:].isdigit():
        return t  # scientific notation like p38, p21 — conventionally lowercase-led
    return t[0].upper() + t[1:]


def title_case(s):
    """Display-only title-casing for compound/MoA labels.

    Lives here (not routes.py) so generate_interpretation() below can title-
    case the MoA names it interpolates into prose without a circular import.
    Only applied at presentation boundaries — never to the underlying data,
    so search matching and moa_descriptions.json lookups (which use the
    original casing) are unaffected. Preserves tokens that already contain an
    uppercase letter (acronyms like HDAC, MAPK, PPAR) and lowercase-led
    scientific notation (p38, p21); capitalizes each hyphen-separated segment
    otherwise (e.g. "non-nucleoside" -> "Non-Nucleoside").
    """
    if not s:
        return s
    words = []
    for w in s.split(" "):
        if "-" in w:
            words.append("-".join(_cap_token(p) for p in w.split("-")))
        else:
            words.append(_cap_token(w))
    return " ".join(words)


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


def compute_neighborhood_stats(query_moa, neighbor_moas, similarities):
    """Pure arithmetic over a neighbor result — the evidence, not the story.

    query_moa: the query's annotated MoA, or None/empty if unannotated.
    neighbor_moas: list of each neighbor's MoA (None/empty allowed), same
        length and order as `similarities`.
    similarities: list/array of cosine similarity scores for those neighbors.

    Returns a dict consumed directly by the sidebar's "Neighborhood Summary"
    section and by generate_interpretation() below — no statistic here is
    invented beyond what the old combined insight paragraph already computed
    internally; this just exposes it. "Neighborhood consistency" buckets the
    dominant-MoA share (already computed) into High (>=60%), Moderate (>=40%),
    or Low — thresholds chosen for readability, not derived from the data.
    """
    if not neighbor_moas:
        return None

    k = len(neighbor_moas)
    avg_sim = float(np.mean(similarities))
    norm_moas = [m if m else "Unannotated" for m in neighbor_moas]
    dominant_moa, dominant_count = Counter(norm_moas).most_common(1)[0]
    dominant_frac = dominant_count / k
    has_query_moa = bool(query_moa)
    shared = sum(1 for m in norm_moas if has_query_moa and m == query_moa)

    if dominant_frac >= 0.6:
        consistency = "High"
    elif dominant_frac >= 0.4:
        consistency = "Moderate"
    else:
        consistency = "Low"

    return {
        "k": k,
        "mean_similarity": avg_sim,
        "has_query_moa": has_query_moa,
        "shared_count": shared,
        "dominant_moa": dominant_moa,
        "dominant_count": dominant_count,
        "dominant_frac": dominant_frac,
        "consistency": consistency,
        "dominant_is_unannotated": dominant_moa == "Unannotated",
        "dominant_matches_query": has_query_moa and dominant_moa == query_moa,
    }


def generate_interpretation(entity_label: str, query_moa, stats) -> str:
    """Cautiously-worded conclusion, built only from the stats dict above.

    No LLM, no external calls, and no new numbers — this is prose around
    figures already shown in the Neighborhood Summary section, phrased as
    evidence/hypothesis language rather than a confirmed conclusion.
    """
    if stats is None:
        return "No neighbors were found to compare against."

    query_moa_display = title_case(query_moa)
    dominant_moa_display = title_case(stats["dominant_moa"])

    if stats["dominant_is_unannotated"] and stats["dominant_frac"] >= 0.5:
        if stats["has_query_moa"]:
            return (
                f"Most neighbors in this phenotypic neighborhood have no annotated mechanism of "
                f"action, which limits how much can be concluded about this {entity_label}'s "
                f"{query_moa_display} annotation from this comparison."
            )
        return (
            "Most neighbors in this phenotypic neighborhood also lack an annotated mechanism of "
            "action, which limits how much can be concluded from this comparison."
        )

    if stats["dominant_matches_query"]:
        return (
            f"This {entity_label}'s phenotypic neighborhood is consistent with its annotated "
            f"mechanism, {query_moa_display}. This is supporting evidence for — not independent "
            "confirmation of — the existing annotation."
        )

    if stats["has_query_moa"]:
        return (
            f"This {entity_label} occupies a neighborhood that is not dominated by other "
            f"{query_moa_display} compounds. This may reflect overlapping cellular phenotypes, "
            "annotation limitations, or limitations of the learned representation. This observation "
            "should be treated as a hypothesis rather than evidence of a shared biological mechanism."
        )

    return (
        f"This {entity_label} has no annotated mechanism of action. Its phenotypic neighborhood is "
        f"most consistent with {dominant_moa_display}, which may indicate a hypothesis for its "
        "mechanism worth further investigation."
    )
