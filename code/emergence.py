"""
Measurement of the emergent language: topographic similarity as a family of six
variants, one per signal set S1-S6, differing only in the signal distance.

See docs/measurement.md for what each set means, the two meaning spaces, the
``_static`` leakage control, and the divergences from the reference MoverScore
implementation.
"""

from collections import Counter

import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from strsimpy.weighted_levenshtein import WeightedLevenshtein

from rapidfuzz.distance import Levenshtein as _Levenshtein
from rapidfuzz.process import cdist as _rf_cdist


_EPS = 1e-12


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
# Meaning-space distances (the same six signal distances are read against each) #
# --------------------------------------------------------------------------- #
def concept_distance_condensed(concepts, metric="cosine"):
    """
    Condensed pairwise distances between sender concept vectors.

    ``concepts`` is the third output of ``Sender.speak`` -- the positive and
    negative prototypes concatenated -- one row per item.
    """
    concepts = np.asarray(concepts, dtype=np.float64)
    return pdist(concepts, metric=metric)


def formula_distance_condensed(formulas):
    """
    Condensed word-level edit distance between ground-truth concept formulas.

    The second meaning space; ``topsim_gt_s6`` read against it is the variant
    comparable to Mu & Goodman (2021). See docs/measurement.md.

    Words are interned to integers so the compiled ``rapidfuzz`` path can be
    used; it is exact on any sequence of hashables.
    """
    vocabulary = {}
    encoded = [
        [vocabulary.setdefault(word, len(vocabulary)) for word in formula.split()]
        for formula in formulas
    ]
    full = _rf_cdist(
        encoded, encoded, scorer=_Levenshtein.distance, dtype=np.float64
    )
    return squareform(full, checks=False)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def _unit_rows(vectors):
    """L2-normalise the rows of a 2-d array, leaving zero rows as zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, _EPS)


def _ngrams(seq, n):
    """The stride-1 n-grams of ``seq`` as a list of token-id tuples."""
    seq = [int(t) for t in seq]
    if len(seq) < n:
        return []
    return [tuple(seq[k:k + n]) for k in range(len(seq) - n + 1)]


def _ngram_vectors(unit_embeddings, n):
    """
    n-gram vectors for one message, from its L2-normalised symbol embeddings.

    For ``n == 1`` these are the unit symbol embeddings themselves. For larger
    ``n`` the constituent unit vectors are *concatenated in order* and the
    result re-normalised -- not summed as in the reference implementation; see
    docs/measurement.md.
    """
    unit_embeddings = np.asarray(unit_embeddings, dtype=np.float64)
    length, dim = unit_embeddings.shape
    if length < n:
        return np.zeros((0, dim * n), dtype=np.float64)
    if n == 1:
        return unit_embeddings
    windows = [unit_embeddings[k:length - n + 1 + k] for k in range(n)]
    return _unit_rows(np.concatenate(windows, axis=1))


def _condensed_shape(n_items):
    return n_items * (n_items - 1) // 2


# --------------------------------------------------------------------------- #
# S1 / S2: soft MoverScore                                                     #
# --------------------------------------------------------------------------- #
def _soft_mover_pair(gram_ids_a, vecs_a, gram_ids_b, vecs_b):
    """
    Raw earth-mover transport cost between two messages' n-gram supports.

    The ground cost is the Euclidean distance between joint-support unit
    vectors, overridden to 0 wherever the two n-gram ids match -- which is what
    preserves Identity under contextual embeddings. See docs/measurement.md.
    """
    m_a, m_b = len(gram_ids_a), len(gram_ids_b)
    if m_a == 0 or m_b == 0:
        # EMD between an empty measure and anything is undefined.
        return np.nan

    raw = np.concatenate([vecs_a, vecs_b], axis=0)
    raw = raw / (np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-6)
    cost = np.sqrt(np.maximum(2.0 - 2.0 * (raw @ raw.T), 1e-30))

    gram_ids = np.concatenate([gram_ids_a, gram_ids_b])
    cost[gram_ids[:, None] == gram_ids[None, :]] = 0.0

    # Uniform mass over positions; IDF is dropped. See docs/measurement.md.
    mass_a = np.zeros(m_a + m_b, dtype=np.float64)
    mass_a[:m_a] = 1.0 / m_a
    mass_b = np.zeros(m_a + m_b, dtype=np.float64)
    mass_b[m_a:] = 1.0 / m_b

    return float(ot.emd2(mass_a, mass_b, np.ascontiguousarray(cost)))


def soft_mover_condensed(token_seqs, symbol_embeddings, n=1):
    """
    Soft MoverScore transport cost over n-grams, condensed (``pdist`` order).

    Signal distance for S1 (``n=1``) and S2 (``n=2``).

    Parameters
    ----------
    token_seqs : list of list of int
        Emergent messages as content-token-id sequences.
    symbol_embeddings : list of array ``(len(seq), d)``
        The sender's contextual embedding for each emitted symbol, positionally
        aligned with ``token_seqs``. May be passed as one ``(N, L, d)`` array
        when all messages are the same length.
    """
    n_items = len(token_seqs)

    # One integer id per distinct n-gram across the whole corpus, so the
    # identity override is a cheap numpy comparison rather than tuple equality.
    gram_vocab = {}
    supports = []
    for seq, emb in zip(token_seqs, symbol_embeddings):
        grams = _ngrams(seq, n)
        ids = np.array(
            [gram_vocab.setdefault(g, len(gram_vocab)) for g in grams],
            dtype=np.int64,
        )
        supports.append((ids, _ngram_vectors(_unit_rows(emb), n)))

    out = np.empty(_condensed_shape(n_items), dtype=np.float64)
    k = 0
    for i in range(n_items - 1):
        ids_i, vecs_i = supports[i]
        for j in range(i + 1, n_items):
            ids_j, vecs_j = supports[j]
            out[k] = _soft_mover_pair(ids_i, vecs_i, ids_j, vecs_j)
            k += 1
    return out


# --------------------------------------------------------------------------- #
# S4 / S5: hard MoverScore                                                     #
# --------------------------------------------------------------------------- #
def hard_mover_condensed(token_seqs, n=1):
    """
    Earth-mover transport cost over n-grams under a 0/1 ground cost, condensed.

    Signal distance for S4 (``n=1``) and S5 (``n=2``).

    Computed in closed form: under a 0/1 ground cost the cost between normalised
    histograms is ``0.5 * ||p - q||_1``. ``tests/test_emergence.py`` asserts this
    against ``ot.emd2``.
    """
    n_items = len(token_seqs)
    gram_vocab = {}
    counts = []
    for seq in token_seqs:
        counter = Counter(_ngrams(seq, n))
        for gram in counter:
            gram_vocab.setdefault(gram, len(gram_vocab))
        counts.append(counter)

    if not gram_vocab:
        return np.full(_condensed_shape(n_items), np.nan)

    hist = np.zeros((n_items, len(gram_vocab)), dtype=np.float64)
    for i, counter in enumerate(counts):
        for gram, count in counter.items():
            hist[i, gram_vocab[gram]] = count

    total = hist.sum(axis=1, keepdims=True)
    # A message with no n-grams (shorter than n) carries no mass; EMD against it
    # is undefined, so propagate NaN rather than inventing a distance.
    hist = np.divide(
        hist, total, out=np.full_like(hist, np.nan), where=total > 0
    )
    return pdist(hist, metric="cityblock") / 2.0


# --------------------------------------------------------------------------- #
# S3: soft Levenshtein                                                         #
# --------------------------------------------------------------------------- #
def soft_levenshtein_condensed(token_seqs, symbol_embeddings):
    """
    Levenshtein distance with a synonymy-tolerant substitution cost, condensed.

    Signal distance for S3. Substituting one symbol for another costs 0 when the
    token ids match and ``1 - cos`` between their contextual embeddings
    otherwise; insertion and deletion cost 1 each.

    Elements handed to strsimpy are ``(utterance_index, position, token_id)``
    tuples so the cost callback can reach the embedding for that position. That
    defeats strsimpy's internal ``s0i != s1j`` short-circuit, which is why the
    token-id equality check lives inside the cost function.
    """
    units = [_unit_rows(emb) for emb in symbol_embeddings]

    def substitution_cost(a, b):
        if a[2] == b[2]:
            return 0.0
        return 1.0 - float(units[a[0]][a[1]] @ units[b[0]][b[1]])

    metric = WeightedLevenshtein(
        substitution_cost_fn=substitution_cost,
        insertion_cost_fn=lambda element: 1.0,
        deletion_cost_fn=lambda element: 1.0,
    )

    tagged = [
        [(i, position, int(token)) for position, token in enumerate(seq)]
        for i, seq in enumerate(token_seqs)
    ]

    n_items = len(token_seqs)
    out = np.empty(_condensed_shape(n_items), dtype=np.float64)
    k = 0
    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            out[k] = metric.distance(tagged[i], tagged[j])
            k += 1
    return out


# --------------------------------------------------------------------------- #
# S6: hard Levenshtein (classic topsim)                                        #
# --------------------------------------------------------------------------- #
def hard_levenshtein_condensed(token_seqs):
    """
    Plain Levenshtein distance between token sequences, condensed.

    Signal distance for S6 -- the classic topsim. Raw edit counts, not
    length-normalised: messages are fixed-length here.
    """
    full = _rf_cdist(
        token_seqs, token_seqs, scorer=_Levenshtein.distance, dtype=np.float64
    )
    return squareform(full, checks=False)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
SIGNAL_SETS = ("s1", "s2", "s3", "s4", "s5", "s6")

# The synonymy-tolerant half; only these read the sender's symbol embeddings,
# and so only these get a `_static` control.
SOFT_SIGNAL_SETS = ("s1", "s2", "s3")

# Against the sender's own concept vectors (cosine)...
TOPSIM_KEYS = tuple(f"topsim_{s}" for s in SIGNAL_SETS)
# ...and against the ground-truth logical forms (word-level edit distance).
TOPSIM_GT_KEYS = tuple(f"topsim_gt_{s}" for s in SIGNAL_SETS)

# Every column `topsim_report` can emit.
REPORT_KEYS = TOPSIM_KEYS + TOPSIM_GT_KEYS + tuple(
    f"{prefix}_{s}_static"
    for prefix in ("topsim", "topsim_gt")
    for s in SOFT_SIGNAL_SETS
)


def soft_signal_distances(token_seqs, symbol_embeddings):
    """The three embedding-parameterised signal distances, keyed by signal set."""
    return {
        "s1": soft_mover_condensed(token_seqs, symbol_embeddings, n=1),
        "s2": soft_mover_condensed(token_seqs, symbol_embeddings, n=2),
        "s3": soft_levenshtein_condensed(token_seqs, symbol_embeddings),
    }


def signal_distances(token_seqs, symbol_embeddings):
    """
    All six condensed signal-distance vectors, keyed by signal set.

    Split out from :func:`topsim_suite` so the distances can be inspected (and
    tested) without a meaning space.
    """
    return {
        **soft_signal_distances(token_seqs, symbol_embeddings),
        "s4": hard_mover_condensed(token_seqs, n=1),
        "s5": hard_mover_condensed(token_seqs, n=2),
        "s6": hard_levenshtein_condensed(token_seqs),
    }


def topsim_suite(token_seqs, symbol_embeddings, concepts):
    """
    The six topsim variants for one set of (message, embeddings, concept)
    triples, against the sender's own concept vectors (cosine).

    Parameters
    ----------
    token_seqs : list of list of int
        Emergent messages as content-token-id sequences.
    symbol_embeddings : list of array ``(len(seq), d)``
        Contextual symbol embeddings, positionally aligned with ``token_seqs``.
    concepts : array ``(n, d_c)``
        Sender concept vectors, one row per message.

    Returns
    -------
    dict[str, float]
        Keys ``topsim_s1`` ... ``topsim_s6``. A value is ``nan`` where Spearman
        is undefined.
    """
    return _correlate(
        signal_distances(token_seqs, symbol_embeddings),
        concept_distance_condensed(concepts),
        "topsim",
    )


def _correlate(signal_conds, meaning_cond, prefix, suffix=""):
    """One meaning distance against a set of signal distances."""
    return {
        f"{prefix}_{name}{suffix}": spearman_topsim(meaning_cond, signal_cond)
        for name, signal_cond in signal_conds.items()
    }


def decontextualised_embeddings(token_seqs, symbol_embeddings):
    """
    Each symbol's contextual embedding replaced by the corpus mean for its id.

    This strips the embeddings of their sensitivity to the concept being
    described while leaving synonymy detection intact, and is what the
    ``_static`` columns are computed on. See docs/measurement.md.
    """
    totals, counts = {}, Counter()
    dim = None
    for seq, embedding in zip(token_seqs, symbol_embeddings):
        embedding = np.asarray(embedding, dtype=np.float64)
        dim = embedding.shape[-1]
        for position, token in enumerate(seq):
            token = int(token)
            if token in totals:
                totals[token] += embedding[position]
            else:
                totals[token] = embedding[position].copy()
            counts[token] += 1

    table = {token: totals[token] / counts[token] for token in totals}
    return [
        np.stack([table[int(token)] for token in seq])
        if len(seq) else np.zeros((0, dim), dtype=np.float64)
        for seq in token_seqs
    ]


def topsim_report(token_seqs, symbol_embeddings, concepts, formulas=None):
    """
    The six variants against both meaning spaces, plus the ``_static`` leakage
    control on the three soft variants. See docs/measurement.md.

    Parameters
    ----------
    token_seqs, symbol_embeddings, concepts
        As :func:`topsim_suite`.
    formulas : list of str, optional
        Ground-truth logical form per utterance. When given, the ``topsim_gt_``
        family is reported alongside. Omit for datasets whose concepts have no
        logical form (CUB).

    Returns
    -------
    dict[str, float]
        A subset of :data:`REPORT_KEYS` -- the ``topsim_gt_`` keys are absent
        when ``formulas`` is ``None``.
    """
    real = signal_distances(token_seqs, symbol_embeddings)
    static = soft_signal_distances(
        token_seqs, decontextualised_embeddings(token_seqs, symbol_embeddings)
    )

    meanings = {"topsim": concept_distance_condensed(concepts)}
    if formulas is not None:
        meanings["topsim_gt"] = formula_distance_condensed(formulas)

    report = {}
    for prefix, meaning_cond in meanings.items():
        report.update(_correlate(real, meaning_cond, prefix))
        report.update(_correlate(static, meaning_cond, prefix, "_static"))
    return report
