"""
Measurement of the emergent language.

Only two things are measured in this codebase: generalisation accuracy (taken
straight off the listener's predictions in ``train.py``) and *topographic
similarity* (topsim) -- the Spearman correlation between pairwise distances in
meaning space and pairwise distances in signal (message) space.

Classic topsim uses a single signal distance, Levenshtein, which is sensitive
to both the order of the symbols and their identity. A language with free
symbol order, or one with synonyms, is therefore scored as non-compositional
even when it is perfectly compositional. This module implements topsim as a
*family* of six variants, one per signal set S1-S6, differing only in the
signal distance function:

    key         set  signal distance             characterised by
    topsim_s1   S1   soft MoverScore, 1-grams    free order + synonymy
    topsim_s2   S2   soft MoverScore, 2-grams    blockwise free order + synonymy
    topsim_s3   S3   soft Levenshtein            strict order + synonymy
    topsim_s4   S4   hard MoverScore, 1-grams    free order, no synonymy
    topsim_s5   S5   hard MoverScore, 2-grams    blockwise free order, no synonymy
    topsim_s6   S6   hard Levenshtein            strict order, no synonymy

S6 is the classic topsim. "Soft" means synonymy-tolerant: the cost of aligning
two symbols is a function of the sender's own contextual symbol embeddings
(``Sender.speak``). "Hard" is the same function under a fixed, embedding-free
0/1 ground cost, so only symbol identity matters.

The *meaning* distance is held constant across all six: cosine distance between
the sender's concept vectors (the output of ``Sender.get_concepts``).

Reference implementations
-------------------------
The MoverScore variants follow ``moverscore.py`` (v1) from
https://github.com/AIPHES/emnlp19-moverscore -- the version with n-gram support
and ``score = 1 - emd(...)``. We take the raw transport cost ``emd(c1, c2, D)``
as our distance, i.e. the quantity that repo subtracts from 1. IDF weighting is
dropped (emergent languages violate Zipf's Law of Abbreviation), so the masses
are uniform over n-gram positions.

Two deliberate divergences from that reference:

1. **2-gram embeddings are concatenated, not summed.** ``load_ngram`` builds an
   n-gram vector as an IDF-weighted *sum* over the window. With IDF dropped
   that degenerates to a plain mean, which is order-blind *within* the n-gram:
   "AB" and "BA" would embed identically, collapsing S2 into S1 and destroying
   the exact distinction the variant exists to draw. We concatenate the two
   unit vectors in order and re-normalise instead, so the resulting cost is a
   monotone function of the mean cosine similarity of the aligned constituents.
2. **Zero ground cost on token match.** Symbol embeddings here are
   *contextual*, so two different games emitting the same token sequence would
   otherwise be at non-zero distance from each other, violating the Identity
   property topsim needs. Wherever the two n-gram ids match, the ground cost is
   overridden to 0. This applies to the soft Levenshtein substitution cost too,
   and it is what guarantees that soft is never more expensive than hard on the
   symbols the two languages agree about.

The Levenshtein variants use ``strsimpy.weighted_levenshtein`` from
https://github.com/luozhouyang/python-string-similarity for the soft case, and
``rapidfuzz`` for the hard case (an exact, compiled equivalent).

Message geometry in this codebase
---------------------------------
Both senders mask the reserved tokens (PAD/SOS/EOS/UNK) out of the content
logits, so EOS never fires early and every message is exactly
``message_length - 2`` content symbols. Insertions and deletions therefore
never arise, length normalisation is a no-op (so raw edit counts are reported),
and a 5-symbol message has only 4 two-grams.
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
# Meaning-space distance (the same for all six variants)                       #
# --------------------------------------------------------------------------- #
def concept_distance_condensed(concepts, metric="cosine"):
    """
    Condensed pairwise distances between sender concept vectors.

    ``concepts`` is the output of ``Sender.get_concepts`` -- the positive and
    negative prototypes concatenated -- one row per item.
    """
    concepts = np.asarray(concepts, dtype=np.float64)
    return pdist(concepts, metric=metric)


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
    result re-normalised -- see the module docstring on why the reference
    implementation's weighted sum is not usable here.
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

    Follows moverscore v1: the two supports are stacked into one joint support,
    the rows re-normalised, and the ground cost taken as the Euclidean distance
    between those unit vectors (``sqrt(2 - 2 cos)``, monotone in cosine
    distance). The cost is then overridden to 0 wherever the two n-gram ids
    match, which is what preserves Identity under contextual embeddings.
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

    # Uniform mass over positions (IDF is dropped -- see the module docstring).
    # The reference's `_safe_divide` zero-guard would only rescale these by a
    # constant, and would rescale the two sides differently when the messages
    # differ in length, so normalise exactly instead.
    mass_a = np.zeros(m_a + m_b, dtype=np.float64)
    mass_a[:m_a] = 1.0 / m_a
    mass_b = np.zeros(m_a + m_b, dtype=np.float64)
    mass_b[m_a:] = 1.0 / m_b

    return float(ot.emd2(mass_a, mass_b, np.ascontiguousarray(cost)))


def soft_mover_condensed(token_seqs, symbol_embeddings, n=1):
    """
    Soft MoverScore transport cost over n-grams, condensed (``pdist`` order).

    Signal distance for S1 (``n=1``, free order + synonymy) and S2 (``n=2``,
    blockwise free order + synonymy).

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

    Signal distance for S4 (``n=1``, free order, no synonymy) and S5 (``n=2``,
    blockwise free order, no synonymy).

    Computed in closed form. Under a 0/1 ground cost the optimal plan matches as
    much mass as possible at zero cost, so for normalised n-gram histograms
    ``p`` and ``q`` the transport cost is ``1 - sum_g min(p_g, q_g)``, which is
    exactly ``0.5 * ||p - q||_1``. ``tests/test_emergence.py`` asserts this
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

    Signal distance for S3 (strict order + synonymy). Substituting one symbol
    for another costs 0 when the token ids match and ``1 - cos`` between their
    contextual embeddings otherwise; insertion and deletion cost 1 each.

    Elements handed to strsimpy are ``(utterance_index, position, token_id)``
    tuples so that the cost callback can reach the contextual embedding for that
    specific position. That defeats strsimpy's internal ``s0i != s1j``
    short-circuit -- which is exactly why the token-id equality check has to
    live inside our cost function.

    Note ``1 - cos <= 2 = insertion + deletion``, so substitution is never
    dominated by a delete/insert pair and the DP stays well-formed. Messages are
    fixed-length here, so in practice neither ever fires.
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

    Signal distance for S6 (strict order, no synonymy) -- the classic topsim.
    Raw edit counts, not length-normalised: messages are fixed-length in this
    setup, so normalising by mean length is a no-op that only obscures the
    metric. ``tests/test_emergence.py`` asserts equality against the strsimpy
    DP that the soft variant uses.
    """
    full = _rf_cdist(
        token_seqs, token_seqs, scorer=_Levenshtein.distance, dtype=np.float64
    )
    return squareform(full, checks=False)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
SIGNAL_SETS = ("s1", "s2", "s3", "s4", "s5", "s6")

TOPSIM_KEYS = tuple(f"topsim_{s}" for s in SIGNAL_SETS)


def signal_distances(token_seqs, symbol_embeddings):
    """
    All six condensed signal-distance vectors, keyed by signal set.

    Split out from :func:`topsim_suite` so the distances can be inspected (and
    tested) without a meaning space.
    """
    return {
        "s1": soft_mover_condensed(token_seqs, symbol_embeddings, n=1),
        "s2": soft_mover_condensed(token_seqs, symbol_embeddings, n=2),
        "s3": soft_levenshtein_condensed(token_seqs, symbol_embeddings),
        "s4": hard_mover_condensed(token_seqs, n=1),
        "s5": hard_mover_condensed(token_seqs, n=2),
        "s6": hard_levenshtein_condensed(token_seqs),
    }


def topsim_suite(token_seqs, symbol_embeddings, concepts):
    """
    The six topsim variants for one set of (message, embeddings, concept) triples.

    Each is the Spearman correlation of one signal set's distance against the
    single meaning distance (cosine between concept vectors). Keys are named by
    *signal set*, not by distance function, because the set is what the value
    licenses a claim about.

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
        is undefined (fewer than two finite pairs, or either side constant).
    """
    meaning_cond = concept_distance_condensed(concepts)
    return {
        f"topsim_{name}": spearman_topsim(meaning_cond, signal_cond)
        for name, signal_cond in signal_distances(
            token_seqs, symbol_embeddings
        ).items()
    }
