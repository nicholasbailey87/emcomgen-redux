"""
Measurement of the emergent language: topographic similarity as a family of six
variants, one per signal set S1-S6, differing only in the signal distance.

Each variant is measured per game, over a seeded random sample of game pairs,
with a 95% bootstrap interval that resamples the games.

See docs/measurement.md for what each set means, the two meaning spaces, the
``_static`` leakage control, the pair sample and its intervals, and the
divergences from the reference MoverScore implementation.
"""

from collections import Counter

import numpy as np
import ot
from scipy.stats import rankdata, spearmanr
from strsimpy.weighted_levenshtein import WeightedLevenshtein

from rapidfuzz.distance import Levenshtein as _Levenshtein


_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Spearman over condensed distance vectors                                     #
# --------------------------------------------------------------------------- #
def spearman_topsim(meaning_cond, message_cond):
    """
    Spearman correlation between two pairwise-distance vectors.

    The vectors may be condensed (every pair, ``pdist`` order) or over a sample
    of pairs; all that matters is that both list the same pairs in the same
    order.

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
# Pairs                                                                        #
# --------------------------------------------------------------------------- #
def _condensed_shape(n_items):
    return n_items * (n_items - 1) // 2


def _condensed_to_pairs(k, n_items):
    """
    Map condensed (``pdist``-order) indices back to ``(i, j)`` with ``i < j``.

    Row ``i`` starts at ``i * (2n - i - 1) / 2``. The closed-form root can land
    one row out in floating point at large ``n``, so it is corrected against
    the exact integer row starts.
    """
    k = np.asarray(k, dtype=np.int64)
    n = int(n_items)

    def row_start(i):
        return i * (2 * n - i - 1) // 2

    i = np.floor(
        (2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8.0 * k)) / 2
    ).astype(np.int64)
    i = np.clip(i, 0, max(n - 2, 0))
    i -= (row_start(i) > k).astype(np.int64)
    i += (row_start(i + 1) <= k).astype(np.int64)
    j = k - row_start(i) + i + 1
    return i, j


def all_pairs(n_items):
    """Every unordered pair, in ``pdist`` order."""
    return _condensed_to_pairs(np.arange(_condensed_shape(n_items)), n_items)


def sample_pairs(n_items, n_pairs, rng):
    """
    ``n_pairs`` distinct unordered pairs of items, drawn uniformly.

    Returned as two index arrays ``(i, j)`` with ``i < j``, sorted into
    ``pdist`` order. When the budget covers every pair, every pair is returned,
    so a small corpus is measured exactly.

    Parameters
    ----------
    rng : ``numpy.random.Generator``
    """
    total = _condensed_shape(n_items)
    if n_pairs >= total:
        return all_pairs(n_items)
    k = np.sort(rng.choice(total, size=n_pairs, replace=False))
    return _condensed_to_pairs(k, n_items)


def _encode_words(formulas):
    """
    Each formula as a list of integer word ids, so the compiled ``rapidfuzz``
    path can be used; it is exact on any sequence of hashables.
    """
    vocabulary = {}
    return [
        [vocabulary.setdefault(word, len(vocabulary)) for word in formula.split()]
        for formula in formulas
    ]


def _levenshtein_pairs(encoded, pairs):
    """Plain Levenshtein distance between sequences, one value per pair."""
    i, j = pairs
    return np.array(
        [_Levenshtein.distance(encoded[a], encoded[b]) for a, b in zip(i, j)],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Meaning-space distances (the same six signal distances are read against each) #
# --------------------------------------------------------------------------- #
def concept_distance_pairs(concepts, pairs):
    """
    Cosine distance between sender concept vectors, one value per pair.

    ``concepts`` is the third output of ``Sender.speak`` -- the positive and
    negative prototypes concatenated -- one row per item. A zero row gives NaN,
    as ``pdist`` does.
    """
    concepts = np.asarray(concepts, dtype=np.float64)
    i, j = pairs
    norms = np.linalg.norm(concepts, axis=1)
    dots = np.einsum("pd,pd->p", concepts[i], concepts[j])
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - dots / (norms[i] * norms[j])


def concept_distance_condensed(concepts):
    """:func:`concept_distance_pairs` over every pair, in ``pdist`` order."""
    return concept_distance_pairs(concepts, all_pairs(len(concepts)))


def formula_distance_pairs(formulas, pairs):
    """
    Word-level edit distance between ground-truth concept formulas, one value
    per pair.

    The second meaning space; ``topsim_gt_s6`` read against it is the variant
    comparable to Mu & Goodman (2021). See docs/measurement.md.
    """
    return _levenshtein_pairs(_encode_words(formulas), pairs)


def formula_distance_condensed(formulas):
    """:func:`formula_distance_pairs` over every pair, in ``pdist`` order."""
    return formula_distance_pairs(formulas, all_pairs(len(formulas)))


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


def soft_mover_pairs(token_seqs, symbol_embeddings, pairs, n=1):
    """
    Soft MoverScore transport cost over n-grams, one value per pair.

    Signal distance for S1 (``n=1``) and S2 (``n=2``).

    Parameters
    ----------
    token_seqs : list of list of int
        Emergent messages as content-token-id sequences.
    symbol_embeddings : list of array ``(len(seq), d)``
        The sender's contextual embedding for each emitted symbol, positionally
        aligned with ``token_seqs``. May be passed as one ``(N, L, d)`` array
        when all messages are the same length.
    pairs : ``(i, j)`` index arrays, as :func:`sample_pairs` returns
    """
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

    i, j = pairs
    return np.array(
        [_soft_mover_pair(*supports[a], *supports[b]) for a, b in zip(i, j)],
        dtype=np.float64,
    )


def soft_mover_condensed(token_seqs, symbol_embeddings, n=1):
    """:func:`soft_mover_pairs` over every pair, in ``pdist`` order."""
    return soft_mover_pairs(
        token_seqs, symbol_embeddings, all_pairs(len(token_seqs)), n=n
    )


# --------------------------------------------------------------------------- #
# S4 / S5: hard MoverScore                                                     #
# --------------------------------------------------------------------------- #
def hard_mover_pairs(token_seqs, pairs, n=1):
    """
    Earth-mover transport cost over n-grams under a 0/1 ground cost, one value
    per pair.

    Signal distance for S4 (``n=1``) and S5 (``n=2``).

    Computed in closed form: under a 0/1 ground cost the cost between normalised
    histograms is ``0.5 * ||p - q||_1``. ``tests/test_emergence.py`` asserts this
    against ``ot.emd2``.
    """
    i, j = pairs
    n_items = len(token_seqs)
    gram_vocab = {}
    counts = []
    for seq in token_seqs:
        counter = Counter(_ngrams(seq, n))
        for gram in counter:
            gram_vocab.setdefault(gram, len(gram_vocab))
        counts.append(counter)

    if not gram_vocab:
        return np.full(len(i), np.nan)

    hist = np.zeros((n_items, len(gram_vocab)), dtype=np.float64)
    for item, counter in enumerate(counts):
        for gram, count in counter.items():
            hist[item, gram_vocab[gram]] = count

    total = hist.sum(axis=1, keepdims=True)
    # A message with no n-grams (shorter than n) carries no mass; EMD against it
    # is undefined, so propagate NaN rather than inventing a distance.
    hist = np.divide(
        hist, total, out=np.full_like(hist, np.nan), where=total > 0
    )
    return np.abs(hist[i] - hist[j]).sum(axis=1) / 2.0


def hard_mover_condensed(token_seqs, n=1):
    """:func:`hard_mover_pairs` over every pair, in ``pdist`` order."""
    return hard_mover_pairs(token_seqs, all_pairs(len(token_seqs)), n=n)


# --------------------------------------------------------------------------- #
# S3: soft Levenshtein                                                         #
# --------------------------------------------------------------------------- #
def soft_levenshtein_pairs(token_seqs, symbol_embeddings, pairs):
    """
    Levenshtein distance with a synonymy-tolerant substitution cost, one value
    per pair.

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
        [(item, position, int(token)) for position, token in enumerate(seq)]
        for item, seq in enumerate(token_seqs)
    ]

    i, j = pairs
    return np.array(
        [metric.distance(tagged[a], tagged[b]) for a, b in zip(i, j)],
        dtype=np.float64,
    )


def soft_levenshtein_condensed(token_seqs, symbol_embeddings):
    """:func:`soft_levenshtein_pairs` over every pair, in ``pdist`` order."""
    return soft_levenshtein_pairs(
        token_seqs, symbol_embeddings, all_pairs(len(token_seqs))
    )


# --------------------------------------------------------------------------- #
# S6: hard Levenshtein (classic topsim)                                        #
# --------------------------------------------------------------------------- #
def hard_levenshtein_pairs(token_seqs, pairs):
    """
    Plain Levenshtein distance between token sequences, one value per pair.

    Signal distance for S6 -- the classic topsim. Raw edit counts, not
    length-normalised: messages are fixed-length here.
    """
    return _levenshtein_pairs(
        [[int(t) for t in seq] for seq in token_seqs], pairs
    )


def hard_levenshtein_condensed(token_seqs):
    """:func:`hard_levenshtein_pairs` over every pair, in ``pdist`` order."""
    return hard_levenshtein_pairs(token_seqs, all_pairs(len(token_seqs)))


def saturated_share(token_seqs, pairs):
    """
    The fraction of ``pairs`` whose S6 distance is as large as it can be.

    Levenshtein distance is at most the longer message's length, so a pair at
    that bound shares nothing an alignment could use. A capped distance cannot
    grade meaning past it: every pair beyond it ties, whatever their meanings.
    A finer partition of the meaning space pulls the cap in even when
    neighbouring regions get similar messages, so a rising share is one way
    ``topsim_s6`` can fall without the language getting less systematic. Pairs
    of two empty messages are left out of both counts. NaN when none remain.
    See docs/measurement.md.
    """
    distances = hard_levenshtein_pairs(token_seqs, pairs)
    lengths = np.array([len(seq) for seq in token_seqs])
    i, j = pairs
    bound = np.maximum(lengths[i], lengths[j])
    counted = bound > 0
    if not counted.any():
        return float("nan")
    return float(np.mean(distances[counted] == bound[counted]))


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

# The bounds of each value's 95% bootstrap interval, suffixed onto its key.
INTERVAL_SUFFIXES = ("_low", "_high")
# ...and every column the bootstrap adds to a value: its bounds, and the
# fraction of replicates that were degenerate and so left out of them.
BOOTSTRAP_SUFFIXES = INTERVAL_SUFFIXES + ("_degenerate",)


def _with_intervals(keys):
    return tuple(
        key + suffix for key in keys for suffix in ("",) + BOOTSTRAP_SUFFIXES
    )


# Every column `topsim_report` can emit: each value followed by its interval
# and its degenerate fraction.
REPORT_KEYS = _with_intervals(
    TOPSIM_KEYS + TOPSIM_GT_KEYS + tuple(
        f"{prefix}_{s}_static"
        for prefix in ("topsim", "topsim_gt")
        for s in SOFT_SIGNAL_SETS
    )
)


def soft_signal_distances(token_seqs, symbol_embeddings, pairs=None):
    """
    The three embedding-parameterised signal distances, keyed by signal set.
    Over ``pairs`` when given, else over every pair in ``pdist`` order.
    """
    if pairs is None:
        pairs = all_pairs(len(token_seqs))
    return {
        "s1": soft_mover_pairs(token_seqs, symbol_embeddings, pairs, n=1),
        "s2": soft_mover_pairs(token_seqs, symbol_embeddings, pairs, n=2),
        "s3": soft_levenshtein_pairs(token_seqs, symbol_embeddings, pairs),
    }


def signal_distances(token_seqs, symbol_embeddings, pairs=None):
    """
    All six signal-distance vectors, keyed by signal set. Over ``pairs`` when
    given, else over every pair in ``pdist`` order.

    Split out from :func:`topsim_suite` so the distances can be inspected (and
    tested) without a meaning space.
    """
    if pairs is None:
        pairs = all_pairs(len(token_seqs))
    return {
        **soft_signal_distances(token_seqs, symbol_embeddings, pairs),
        "s4": hard_mover_pairs(token_seqs, pairs, n=1),
        "s5": hard_mover_pairs(token_seqs, pairs, n=2),
        "s6": hard_levenshtein_pairs(token_seqs, pairs),
    }


def topsim_suite(token_seqs, symbol_embeddings, concepts):
    """
    The six topsim variants for one set of (message, embeddings, concept)
    triples, against the sender's own concept vectors (cosine), over every pair
    and without intervals.

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
    meaning = concept_distance_condensed(concepts)
    return {
        f"topsim_{name}": spearman_topsim(meaning, signal)
        for name, signal in signal_distances(token_seqs, symbol_embeddings).items()
    }


def decontextualised_embeddings(token_seqs, symbol_embeddings):
    """
    Each symbol's contextual embedding replaced by the corpus mean for its id
    *in that position*.

    This strips the embeddings of their sensitivity to the concept being
    described while leaving synonymy detection intact, and is what the
    ``_static`` columns are computed on. The mean is taken per (position,
    symbol) rather than per symbol because emergent languages reuse a symbol to
    mean different things in different slots (Chaabouni et al. 2020): pooling
    across positions would blur those meanings together and penalise the
    language for its own positional code. A slot carries nothing about the game,
    so the control still removes leakage. Before 2026-10-11 the mean was per
    symbol. See docs/measurement.md.
    """
    totals, counts = {}, Counter()
    dim = None
    for seq, embedding in zip(token_seqs, symbol_embeddings):
        embedding = np.asarray(embedding, dtype=np.float64)
        dim = embedding.shape[-1]
        for position, token in enumerate(seq):
            cell = (position, int(token))
            if cell in totals:
                totals[cell] += embedding[position]
            else:
                totals[cell] = embedding[position].copy()
            counts[cell] += 1

    table = {cell: totals[cell] / counts[cell] for cell in totals}
    return [
        np.stack([
            table[(position, int(token))] for position, token in enumerate(seq)
        ])
        if len(seq) else np.zeros((0, dim), dtype=np.float64)
        for seq in token_seqs
    ]


# --------------------------------------------------------------------------- #
# Bootstrap intervals over games                                               #
# --------------------------------------------------------------------------- #
def game_weights(n_items, replicates, rng):
    """
    How many times each game is drawn in each bootstrap replicate:
    ``Multinomial(n, 1/n)`` per row, shape ``(replicates, n_items)``.

    Drawn once per report and shared by every column in it, so the intervals of
    two columns describe the same resampled corpora.
    """
    return rng.multinomial(
        n_items, np.full(n_items, 1.0 / n_items), size=replicates
    )


def _weighted_pearson(weights, x, y):
    """
    Pearson correlation of ``x`` and ``y`` under each row of ``weights``.

    ``weights`` is ``(B, P)``; ``x`` and ``y`` are ``(P,)`` and should be
    centred, which keeps the raw-moment formula clear of cancellation. A row
    with no weight or no variance on either side gives NaN.
    """
    total = weights.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = weights @ x / total
        mean_y = weights @ y / total
        var_x = weights @ (x * x) / total - mean_x ** 2
        var_y = weights @ (y * y) / total - mean_y ** 2
        cov = weights @ (x * y) / total - mean_x * mean_y
        rho = cov / np.sqrt(var_x * var_y)
    # Variances that are zero up to rounding are zero.
    degenerate = (total <= 0) | (var_x <= 1e-12 * (x * x).max()) | (
        var_y <= 1e-12 * (y * y).max()
    )
    rho[degenerate] = np.nan
    return rho


def game_bootstrap(meaning, signals, pairs, weights, level=0.95, chunk=100):
    """
    Spearman of each signal distance against ``meaning``, with a percentile
    bootstrap interval that resamples the *games*.

    Resampling the games with replacement leaves the sampled pairs where they
    are and gives each pair ``(i, j)`` the weight ``w_i * w_j``, where ``w`` is
    the replicate's row of ``weights`` (:func:`game_weights`). So one replicate's
    rho is the weighted Pearson correlation of the pairs' ranks. The ranks are
    those of the original sample rather than re-ranked per replicate -- an
    approximation that ``tests/test_emergence.py`` bounds against exact
    resampling -- which is what lets every replicate be one matrix product.
    Pairs between two copies of one game are not in the sample and are not
    added; they would sit at distance zero on both sides.

    A replicate is degenerate when Spearman is undefined on it: no weight left,
    or every weighted pair at one value on either side. It is left out of the
    percentiles, so the interval is conditional on the replicates that are
    not, and the fraction left out is returned with it. In a corpus where one
    game's message differs from all the others, every replicate that does not
    draw that game is degenerate, about ``(1 - 1/n)^n`` of them.

    Parameters
    ----------
    meaning : array ``(P,)``
        Meaning distance per sampled pair.
    signals : dict[str, array ``(P,)``]
        Signal distances per sampled pair, keyed by column.
    pairs : ``(i, j)`` index arrays, as :func:`sample_pairs` returns
    weights : int array ``(B, n_items)``
    level : float
        Coverage of the interval.
    chunk : int
        Replicates processed at once; bounds memory at ``chunk x P`` floats.

    Returns
    -------
    dict[str, tuple[float, float, float, float]]
        ``(rho, low, high, degenerate)`` per column, in the order of
        ``signals``. ``degenerate`` is the fraction of replicates left out;
        where it is 1 the bounds are NaN. Where Spearman is undefined on the
        original sample, ``rho`` and the bounds are NaN and ``degenerate`` is 1:
        a replicate only reweights the sample's finite pairs, so it is
        undefined whenever the sample is, and its replicates are not computed.
    """
    meaning = np.asarray(meaning, dtype=np.float64)
    i, j = pairs
    tail = 100.0 * (1.0 - level) / 2.0

    # Every column starts undefined and keeps its place in `signals` when
    # filled in, so the report's key order does not depend on which are NaN.
    nan = float("nan")
    report = {name: (nan, nan, nan, 1.0) for name in signals}

    # Ranks are taken once per column, over that column's finite pairs. Pairs
    # that are not finite get weight zero in every replicate.
    columns = {}
    for name, signal in signals.items():
        signal = np.asarray(signal, dtype=np.float64)
        rho = spearman_topsim(meaning, signal)
        if np.isnan(rho):
            continue
        finite = np.isfinite(meaning) & np.isfinite(signal)
        x = np.zeros(len(meaning))
        y = np.zeros(len(meaning))
        x[finite] = rankdata(meaning[finite])
        y[finite] = rankdata(signal[finite])
        centre = (finite.sum() + 1) / 2.0
        x[finite] -= centre
        y[finite] -= centre
        columns[name] = (rho, finite, x, y)

    if not columns:
        return report

    replicates = {name: [] for name in columns}
    for start in range(0, len(weights), chunk):
        block = np.asarray(weights[start:start + chunk], dtype=np.float64)
        pair_weights = block[:, i] * block[:, j]
        for name, (_, finite, x, y) in columns.items():
            replicates[name].append(
                _weighted_pearson(pair_weights * finite, x, y)
            )

    for name, (rho, *_) in columns.items():
        draws = np.concatenate(replicates[name])
        defined = np.isfinite(draws)
        degenerate = np.mean(~defined)
        if not defined.any():
            low = high = nan
        else:
            low, high = np.percentile(draws[defined], [tail, 100.0 - tail])
        report[name] = (rho, float(low), float(high), float(degenerate))
    return report


def topsim_report(
    token_seqs,
    symbol_embeddings,
    concepts,
    formulas=None,
    *,
    pairs=None,
    replicates=1000,
    rng=None,
):
    """
    The six variants against both meaning spaces, plus the ``_static`` leakage
    control on the three soft variants, each with a 95% bootstrap interval over
    games. See docs/measurement.md.

    Parameters
    ----------
    token_seqs, symbol_embeddings, concepts
        As :func:`topsim_suite`, one item per game.
    formulas : list of str, optional
        Ground-truth logical form per game. When given, the ``topsim_gt_``
        family is reported alongside. Omit for datasets whose concepts have no
        logical form (CUB).
    pairs : ``(i, j)`` index arrays, optional
        The game pairs to measure, as :func:`sample_pairs` returns. Every pair
        when omitted.
    replicates : int
        Bootstrap replicates behind each interval.
    rng : ``numpy.random.Generator``, optional
        Draws the bootstrap's game multiplicities. Seeded at 0 when omitted.

    Returns
    -------
    dict[str, float]
        A subset of :data:`REPORT_KEYS` -- the ``topsim_gt_`` keys are absent
        when ``formulas`` is ``None``. Each value ``key`` comes with
        ``key_low``, ``key_high`` and ``key_degenerate``, as
        :func:`game_bootstrap` returns them.
    """
    n_items = len(token_seqs)
    if pairs is None:
        pairs = all_pairs(n_items)
    if rng is None:
        rng = np.random.default_rng(0)

    real = signal_distances(token_seqs, symbol_embeddings, pairs)
    static = soft_signal_distances(
        token_seqs,
        decontextualised_embeddings(token_seqs, symbol_embeddings),
        pairs,
    )
    signals = {
        **real,
        **{f"{name}_static": signal for name, signal in static.items()},
    }

    meanings = {"topsim": concept_distance_pairs(concepts, pairs)}
    if formulas is not None:
        meanings["topsim_gt"] = formula_distance_pairs(formulas, pairs)

    weights = game_weights(n_items, replicates, rng)

    report = {}
    for prefix, meaning in meanings.items():
        for name, (rho, low, high, degenerate) in game_bootstrap(
            meaning, signals, pairs, weights
        ).items():
            key = f"{prefix}_{name}"
            report[key] = rho
            report[f"{key}_low"] = low
            report[f"{key}_high"] = high
            report[f"{key}_degenerate"] = degenerate
    return report
