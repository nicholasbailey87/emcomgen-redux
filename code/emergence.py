"""
Tools for measuring emergence of communication.

Topsim (topographic similarity) is the Spearman correlation between pairwise
distances in *meaning* space and pairwise distances in *message* space. The
expensive part historically was an O(n^2) pure-Python ``pdist`` over edit
distance; everything here is built around precomputed *condensed* pairwise
distance vectors (the upper triangle, as returned by ``scipy``'s ``pdist`` /
``squareform``) so the distance math stays vectorized.

Two meaning bases (``ts`` = concept distance, ``reprts`` = sender-representation
cosine) are crossed with three message distances (``lev``, ``tanimoto``,
``sif``) by :func:`topsim_grid`.
"""

import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from rapidfuzz.process import cdist as _rf_cdist
from rapidfuzz.distance import Levenshtein as _Levenshtein
from tiny_lang_embed import build_embeddings


# --------------------------------------------------------------------------- #
# Spearman over condensed distance vectors                                     #
# --------------------------------------------------------------------------- #
def spearman_topsim(meaning_cond, message_cond):
    """
    Spearman correlation between two condensed pairwise-distance vectors.

    NaNs (e.g. cosine distance involving an all-zero/empty message) are dropped
    pairwise. Returns ``nan`` if fewer than two finite pairs remain or if either
    side is constant (Spearman undefined).
    """
    meaning_cond = np.asarray(meaning_cond, dtype=np.float64)
    message_cond = np.asarray(message_cond, dtype=np.float64)
    if meaning_cond.shape != message_cond.shape:
        raise ValueError(
            f"distance vectors differ in length: "
            f"{meaning_cond.shape} vs {message_cond.shape}"
        )

    finite = np.isfinite(meaning_cond) & np.isfinite(message_cond)
    if finite.sum() < 2:
        return float("nan")
    m = meaning_cond[finite]
    s = message_cond[finite]
    # Spearman is undefined when either input has zero variance.
    if np.ptp(m) == 0 or np.ptp(s) == 0:
        return float("nan")
    return float(spearmanr(m, s).correlation)


# --------------------------------------------------------------------------- #
# Meaning-space distances                                                      #
# --------------------------------------------------------------------------- #
def vector_condensed(vectors, metric="cosine"):
    """Condensed pairwise distances over a stack of vectors (scipy ``pdist``)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return pdist(vectors, metric=metric)


def condensed_from_index(dist_matrix, index):
    """
    Expand a small ``K x K`` concept-distance matrix to a condensed pairwise
    vector over ``n`` items, where ``index[i]`` is the concept of item ``i``.

    This is the fast path for concept distances: build the dense ``K x K`` table
    once (K = number of unique concepts, typically a few hundred), then index
    into it rather than looping over all ``n^2`` item pairs. The matrix must have
    a zero diagonal (same concept -> zero distance).
    """
    index = np.asarray(index)
    full = np.asarray(dist_matrix, dtype=np.float64)[np.ix_(index, index)]
    return squareform(full, checks=False)


def lookup_condensed(keys, distance_fn):
    """
    Condensed pairwise distances over arbitrary keys via a Python callable.

    Used for concept distances that come from a precomputed lookup (e.g.
    shapeworld's pairwise Hausdorff table). The loop is O(n^2) but does only
    cheap dict-style lookups -- it was never the bottleneck; the edit-distance
    message side was.
    """
    n = len(keys)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        ki = keys[i]
        for j in range(i + 1, n):
            out[k] = distance_fn(ki, keys[j])
            k += 1
    return out


# --------------------------------------------------------------------------- #
# Message-space distances                                                      #
# --------------------------------------------------------------------------- #
def levenshtein_condensed(token_seqs):
    """
    Length-normalized Levenshtein distance between token sequences, condensed.

    Normalization matches the historical metric: raw edit distance divided by
    the mean of the two sequence lengths. This per-pair rescaling changes ranks
    (and therefore topsim), so it must be reproduced exactly, not swapped for a
    max-length normalization.
    """
    n = len(token_seqs)
    raw = _rf_cdist(token_seqs, token_seqs, scorer=_Levenshtein.distance,
                    dtype=np.float64)
    lens = np.array([max(len(s), 1) for s in token_seqs], dtype=np.float64)
    denom = (lens[:, None] + lens[None, :]) / 2.0
    norm = raw / denom
    return squareform(norm, checks=False)


def _bag_of_symbols(token_seqs, vocab_size):
    X = np.zeros((len(token_seqs), vocab_size), dtype=np.float64)
    for i, s in enumerate(token_seqs):
        for t in s:
            if 0 <= t < vocab_size:
                X[i, t] += 1.0
    return X


def tanimoto_condensed(token_seqs, vocab_size):
    """
    Tanimoto (extended Jaccard) distance over bag-of-symbol count vectors.

    T(a, b) = <a, b> / (|a|^2 + |b|^2 - <a, b>); distance = 1 - T. Order-
    insensitive, complementing the order-sensitive Levenshtein distance. Two
    empty messages are treated as identical (distance 0).
    """
    X = _bag_of_symbols(token_seqs, vocab_size)
    gram = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    denom = sq[:, None] + sq[None, :] - gram
    sim = np.where(denom > 0, gram / np.where(denom > 0, denom, 1.0), 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False)


def _safe_build_embeddings(token_seqs, vocab_size, embedding_dim, window):
    """
    Build per-symbol embeddings, clamping ``embedding_dim`` to the available
    SVD rank.

    ``tiny_lang_embed`` raises ``ValueError`` when ``embedding_dim`` exceeds the
    number of available singular values, and that rank is bounded by *this*
    corpus -- emergent vocab is tiny (~10-20) and early-training language is
    often degenerate (a single repeated token => near rank-0 co-occurrence). We
    parse the available count out of the error and retry, returning ``None`` if
    the corpus cannot support even a 1-d embedding (caller emits NaN).
    """
    dim = int(embedding_dim)
    while dim >= 1:
        try:
            emb = build_embeddings(
                token_seqs, vocab_size=vocab_size, embedding_dim=dim, window=window
            )
            return np.stack([np.asarray(emb[t], dtype=np.float64)
                             for t in range(vocab_size)])
        except ValueError as exc:
            match = re.search(r"available (\d+)", str(exc))
            avail = int(match.group(1)) if match else dim - 1
            new_dim = min(dim - 1, avail)
            if new_dim >= dim:  # no progress; bail rather than loop forever
                return None
            dim = new_dim
    return None


def _first_principal_component(sentence_vectors):
    """First right singular vector of the sentence-embedding matrix (SIF)."""
    if sentence_vectors.shape[0] < 2:
        return None
    if not np.isfinite(sentence_vectors).all():
        return None
    if np.allclose(sentence_vectors, 0.0):
        return None
    # SIF (Arora et al.) removes the top singular direction without centering.
    _, _, vt = np.linalg.svd(sentence_vectors, full_matrices=False)
    return vt[0]


def sif_condensed(
    token_seqs,
    vocab_size,
    embedding_dim=4,
    window=2,
    a=1e-3,
    pc=None,
    return_pc=False,
):
    """
    Cosine distance between SIF sentence embeddings, condensed.

    Symbol embeddings come from ``tiny_lang_embed`` (count-based, deterministic).
    Each message is the frequency-weighted average a/(a+p(w)) of its symbol
    vectors; the first principal component is then removed. ``pc`` lets a caller
    reuse a principal component estimated on a larger (image-level) set for a
    smaller (concept-prototype) set, where estimating it fresh would be noisy.

    Returns a condensed vector of cosine distances (NaN entries where a message
    is empty), or all-NaN if the corpus is too degenerate to embed. When
    ``return_pc`` is true, also returns the principal component used (or None).
    """
    n = len(token_seqs)
    n_pairs = n * (n - 1) // 2
    nan_out = np.full(n_pairs, np.nan)

    counts = np.zeros(vocab_size, dtype=np.float64)
    for s in token_seqs:
        for t in s:
            if 0 <= t < vocab_size:
                counts[t] += 1.0
    total = counts.sum()
    if total == 0:
        return (nan_out, None) if return_pc else nan_out
    weights = a / (a + counts / total)  # per-symbol SIF weight

    emb = _safe_build_embeddings(token_seqs, vocab_size, embedding_dim, window)
    if emb is None:
        return (nan_out, None) if return_pc else nan_out
    dim = emb.shape[1]

    sentences = np.zeros((n, dim), dtype=np.float64)
    for i, s in enumerate(token_seqs):
        toks = [t for t in s if 0 <= t < vocab_size]
        if not toks:
            continue  # empty message -> zero vector -> NaN cosine distances
        weighted = emb[toks] * weights[toks][:, None]
        sentences[i] = weighted.sum(0) / len(toks)

    if pc is None:
        pc = _first_principal_component(sentences)
    if pc is not None:
        sentences = sentences - np.outer(sentences @ pc, pc)

    cond = pdist(sentences, metric="cosine")
    return (cond, pc) if return_pc else cond


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def topsim_grid(
    token_seqs,
    reprs,
    vocab_size,
    concept_meaning_cond=None,
    sif_dim=4,
    sif_window=2,
    sif_pc=None,
):
    """
    Compute the full grid of topsim values for one set of (message, repr) pairs.

    Crosses two meaning bases -- ``ts`` (concept distance, only if
    ``concept_meaning_cond`` is supplied) and ``reprts`` (cosine distance over
    ``reprs``) -- with three message distances (``lev``, ``tanimoto``, ``sif``).

    Parameters
    ----------
    token_seqs : list of list of int
        Emergent messages as token-id sequences.
    reprs : array (n, d)
        Sender representation vectors (``prototypes_concat``), one per message.
    vocab_size : int
        Symbol vocabulary size (for bag-of-symbols and SIF).
    concept_meaning_cond : array, optional
        Precomputed condensed concept-distance vector aligned with ``token_seqs``.
    sif_pc : array, optional
        Principal component to reuse for the SIF distance (see ``sif_condensed``).

    Returns
    -------
    metrics : dict[str, float]
        Keys ``"{base}_{msg}"`` e.g. ``"ts_lev"``, ``"reprts_sif"``.
    sif_pc : array or None
        The SIF principal component computed here (so a caller can reuse it).
    """
    message_conds = {
        "lev": levenshtein_condensed(token_seqs),
        "tanimoto": tanimoto_condensed(token_seqs, vocab_size),
    }
    sif_cond, sif_pc = sif_condensed(
        token_seqs, vocab_size, sif_dim, sif_window, pc=sif_pc, return_pc=True
    )
    message_conds["sif"] = sif_cond

    meaning_conds = {"reprts": vector_condensed(reprs, metric="cosine")}
    if concept_meaning_cond is not None:
        meaning_conds["ts"] = np.asarray(concept_meaning_cond, dtype=np.float64)

    metrics = {}
    for base, mean_cond in meaning_conds.items():
        for msg, msg_cond in message_conds.items():
            metrics[f"{base}_{msg}"] = spearman_topsim(mean_cond, msg_cond)
    return metrics, sif_pc


# --------------------------------------------------------------------------- #
# Other (enumerable) compositionality measures -- unchanged                    #
# --------------------------------------------------------------------------- #
def normalize(ctr):
    total = sum(ctr.values())
    return Counter({k: v / total for k, v in ctr.items()})


def context_independence(concepts, messages):
    r"""
    Measure context independence between concepts c and messages m.

    Let p_cm(c | m) be the conditional probability of context c given message m and
    p_mc(m | c) be the condiational probability of message m given context c
    (we can estimate these by simply enumerating).

    Then for any concept c, we define the "ideal" message m^c as argmax_m
    p_cm(c | m) (i.e., whichever message has the highest conditional
    probability that we are referring to concept c).

    Then,

    CI(concepts, messages) = 1 / len(concepts) \sum_c p_mc(m^c | c) * p_cm(c | m^c).
    """
    p_cm = defaultdict(Counter)
    p_mc = defaultdict(Counter)

    for c, m in zip(concepts, messages):
        # I.e., given message m, conditional probability of c.
        p_cm[m][c] += 1
        p_mc[c][m] += 1

    p_cm = {k: normalize(v) for k, v in p_cm.items()}
    p_mc = {k: normalize(v) for k, v in p_mc.items()}

    # Find ideal messages
    unique_concepts = list(set(concepts))
    unique_messages = list(set(messages))
    cis = []

    for c in unique_concepts:
        mc = None
        best_p_cm = 0.0
        for m in unique_messages:
            this_p_cm = p_cm[m][c]
            if this_p_cm > best_p_cm:
                mc = m
                best_p_cm = this_p_cm
        if mc is None:
            raise RuntimeError(f"Couldn't find ideal concept for {c}")

        ci = p_mc[c][mc] * p_cm[mc][c]

        cis.append(ci)

    return np.mean(cis)


def mutual_information(concepts, messages):
    r"""
    Measure mutual information between concepts c and messages m (assuming
    enumerability)
    """
    from sklearn.metrics import normalized_mutual_info_score

    # Assign int values
    c2i = {}
    m2i = {}

    for c, m in zip(concepts, messages):
        if c not in c2i:
            c2i[c] = len(c2i)
        if m not in m2i:
            m2i[m] = len(m2i)

    cis = [c2i[c] for c in concepts]
    mis = [m2i[m] for m in messages]

    return normalized_mutual_info_score(cis, mis)
