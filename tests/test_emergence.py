"""
Tests for the six topsim variants in code/emergence.py.

Runnable without pytest:  python tests/test_emergence.py

Four things are checked here. First, that each of the six signal distances is a
*valid signal distance function* in the chapter's sense: Identity (an utterance
is at distance zero from itself) and Smoothness (in expectation the distance is
monotone in the number of corrupting symbol substitutions). Second, that the
closed-form "hard" fast paths are numerically equal to the reference libraries
they stand in for -- ``strsimpy`` for Levenshtein and ``ot.emd2`` for the movers
-- which is what keeps "follows the reference implementation" honest. Third,
that the suite actually discriminates the signal sets it claims to: three
hand-built toy languages, differing only in symbol order freedom and synonymy,
are scored the way the chapter says they should be.

Fourth, the two guards on how a reading may be interpreted. The ``_static``
control has to remove the correlation a soft variant reports on a language that
carries no information at all, *and* leave a genuinely compositional language
untouched -- a control that only did the first would subtract away real results.
And the ``topsim_gt_`` family, measured against the ground-truth logical forms
rather than the sender's own concept vectors, has to notice a sender that has
collapsed onto a subset of the visual features.
"""

import os
import sys

import numpy as np
import ot
from strsimpy.weighted_levenshtein import WeightedLevenshtein

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import emergence as E  # noqa: E402


SETS = ("s1", "s2", "s3", "s4", "s5", "s6")

# Content symbols start at 4 in this codebase (0-3 are PAD/SOS/EOS/UNK), and
# every message is exactly `message_length - 2` = 5 symbols.
FIRST_SYMBOL = 4
LENGTH = 5


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _pair_distances(seqs, embs):
    """All six condensed signal distances for one small corpus."""
    return E.signal_distances(seqs, embs)


def _table_embeddings(seqs, table):
    """Per-message embeddings looked up from a context-free symbol table."""
    return [table[np.asarray(s)] for s in seqs]


def _random_table(rng, vocab, dim=32, nonneg=False):
    table = np.abs(rng.randn(vocab, dim)) if nonneg else rng.randn(vocab, dim)
    return table / np.linalg.norm(table, axis=1, keepdims=True)


def _substitute(rng, seq, k, vocab_ids):
    """Corrupt ``seq`` with exactly ``k`` random symbol substitutions."""
    seq = list(seq)
    for pos in rng.choice(len(seq), size=k, replace=False):
        choices = [v for v in vocab_ids if v != seq[pos]]
        seq[pos] = int(rng.choice(choices))
    return seq


def _hard_emd_reference(ids_a, ids_b):
    """
    Earth-mover distance under a 0/1 ground cost, computed the long way.

    Same joint-support construction as the soft mover, so this is the thing
    ``hard_mover_condensed``'s ``0.5 * L1`` closed form has to match.
    """
    ids = list(ids_a) + list(ids_b)
    cost = np.array(
        [[0.0 if x == y else 1.0 for y in ids] for x in ids], dtype=np.float64
    )
    mass_a = np.zeros(len(ids))
    mass_a[: len(ids_a)] = 1.0 / len(ids_a)
    mass_b = np.zeros(len(ids))
    mass_b[len(ids_a):] = 1.0 / len(ids_b)
    return float(ot.emd2(mass_a, mass_b, np.ascontiguousarray(cost)))


# --------------------------------------------------------------------------- #
# Identity                                                                     #
# --------------------------------------------------------------------------- #
def test_identity_all_six():
    """d(x, x) == 0 for every variant, even when the embeddings differ.

    Items 0 and 1 emit the same token sequence from *different* contextual
    embeddings -- two different games that happened to say the same thing. The
    token-match override is what keeps their distance at zero.
    """
    rng = np.random.RandomState(0)
    shared = [4, 7, 5, 9, 7]
    seqs = [shared, list(shared), [6, 6, 8, 4, 5], [9, 4, 4, 7, 8]]
    embs = [rng.randn(LENGTH, 16) for _ in seqs]

    dists = _pair_distances(seqs, embs)
    for name in SETS:
        # pdist order: pair (0, 1) is index 0.
        assert abs(dists[name][0]) < 1e-9, f"{name}: d(x, x) = {dists[name][0]}"


# --------------------------------------------------------------------------- #
# Soft vs hard                                                                 #
# --------------------------------------------------------------------------- #
def test_soft_levenshtein_at_most_hard():
    """
    Soft Levenshtein <= hard Levenshtein.

    The substitution cost is 0 on a token match and ``1 - cos`` otherwise, so
    this holds whenever the symbol embeddings are pairwise non-negatively
    correlated (``1 - cos <= 1``, the hard substitution cost). Embeddings drawn
    from the non-negative orthant satisfy that. It is *not* a guarantee for
    arbitrary embeddings: ``1 - cos`` ranges up to 2, and the chapter's soft
    Levenshtein is defined as cosine distance, which we do not clip.
    """
    rng = np.random.RandomState(1)
    vocab = list(range(FIRST_SYMBOL, FIRST_SYMBOL + 14))
    seqs = [list(rng.choice(vocab, size=LENGTH)) for _ in range(12)]
    table = _random_table(rng, FIRST_SYMBOL + 14, nonneg=True)
    embs = _table_embeddings(seqs, table)

    soft = E.soft_levenshtein_condensed(seqs, embs)
    hard = E.hard_levenshtein_condensed(seqs)
    assert (soft <= hard + 1e-9).all(), f"soft exceeded hard: {soft - hard}"


# --------------------------------------------------------------------------- #
# Smoothness                                                                   #
# --------------------------------------------------------------------------- #
def test_smoothness_monotone_in_substitutions():
    """
    In expectation each distance is monotone in the number of substitutions.

    Insertions and deletions cannot occur here -- the senders mask the reserved
    tokens out of the content logits, so every message is exactly LENGTH
    symbols -- so substitution is the only corrupting edit to test.

    Monotonicity is asserted against sampling error rather than exactly, because
    the 2-gram variants saturate: a 5-symbol message has only 4 two-grams, so by
    k = 4 substitutions every one of them is already disturbed and k = 4 and
    k = 5 are genuinely tied in expectation.
    """
    rng = np.random.RandomState(2)
    vocab = list(range(FIRST_SYMBOL, FIRST_SYMBOL + 14))
    table = _random_table(rng, FIRST_SYMBOL + 14)
    trials = 300

    samples = {name: [] for name in SETS}
    for k in range(LENGTH + 1):
        at_k = {name: [] for name in SETS}
        for _ in range(trials):
            clean = list(rng.choice(vocab, size=LENGTH))
            corrupt = _substitute(rng, clean, k, vocab)
            # A third message keeps the corpus big enough for a condensed
            # vector; only the (0, 1) pair is read.
            seqs = [clean, corrupt, list(rng.choice(vocab, size=LENGTH))]
            dists = _pair_distances(seqs, _table_embeddings(seqs, table))
            for name in SETS:
                at_k[name].append(dists[name][0])
        for name in SETS:
            samples[name].append(np.asarray(at_k[name]))

    for name in SETS:
        curve = [s.mean() for s in samples[name]]
        errors = [s.std(ddof=1) / np.sqrt(len(s)) if s.std() > 0 else 0.0
                  for s in samples[name]]
        for k in range(1, len(curve)):
            tolerance = 3.0 * (errors[k] + errors[k - 1])
            assert curve[k] >= curve[k - 1] - tolerance, (
                f"{name} not monotone in substitutions: "
                f"{[round(c, 4) for c in curve]}"
            )
        assert curve[-1] > curve[0], f"{name} flat under corruption: {curve}"


# --------------------------------------------------------------------------- #
# The hard fast paths agree with the reference libraries                       #
# --------------------------------------------------------------------------- #
def test_hard_levenshtein_matches_strsimpy():
    """The rapidfuzz fast path equals the strsimpy DP the soft variant uses."""
    rng = np.random.RandomState(3)
    vocab = list(range(FIRST_SYMBOL, FIRST_SYMBOL + 14))
    seqs = [list(rng.choice(vocab, size=rng.randint(1, 8))) for _ in range(15)]

    reference_metric = WeightedLevenshtein(
        substitution_cost_fn=lambda a, b: 1.0,
        insertion_cost_fn=lambda c: 1.0,
        deletion_cost_fn=lambda c: 1.0,
    )
    reference = [
        reference_metric.distance(seqs[i], seqs[j])
        for i in range(len(seqs) - 1)
        for j in range(i + 1, len(seqs))
    ]
    fast = E.hard_levenshtein_condensed(seqs)
    assert np.allclose(fast, reference), f"{fast} vs {reference}"


def test_hard_mover_matches_emd():
    """The 0.5 * L1 closed form equals an explicit ot.emd2 solve, n = 1 and 2."""
    rng = np.random.RandomState(4)
    vocab = list(range(FIRST_SYMBOL, FIRST_SYMBOL + 6))  # small: force repeats
    seqs = [list(rng.choice(vocab, size=LENGTH)) for _ in range(15)]

    for n in (1, 2):
        grams = [
            [tuple(s[k:k + n]) for k in range(len(s) - n + 1)] for s in seqs
        ]
        reference = [
            _hard_emd_reference(grams[i], grams[j])
            for i in range(len(seqs) - 1)
            for j in range(i + 1, len(seqs))
        ]
        fast = E.hard_mover_condensed(seqs, n=n)
        assert np.allclose(fast, reference, atol=1e-9), f"n={n}: {fast - reference}"


# --------------------------------------------------------------------------- #
# Order sensitivity per signal set                                             #
# --------------------------------------------------------------------------- #
def test_order_sensitivity_by_set():
    """
    S1/S4 ignore symbol order; S2/S5 tolerate block swaps more than within-block
    reordering; S3/S6 object to both.
    """
    rng = np.random.RandomState(5)
    base = [4, 5, 6, 7, 8]
    # Blocks [4 5][6 7 8] swapped -- only the 2-grams straddling the boundary
    # move. Within-block reorder swaps 4 and 5, breaking 2-grams inside a block.
    block_swap = [6, 7, 8, 4, 5]
    within_block = [5, 4, 6, 7, 8]
    seqs = [base, block_swap, within_block]
    table = _random_table(rng, 16)
    dists = _pair_distances(seqs, _table_embeddings(seqs, table))

    # pdist order over 3 items: 0 = (base, block_swap), 1 = (base, within_block)
    for name in ("s1", "s4"):
        assert abs(dists[name][0]) < 1e-9, f"{name} not order-invariant"
        assert abs(dists[name][1]) < 1e-9, f"{name} not order-invariant"

    for name in ("s2", "s5"):
        assert dists[name][0] > 0, f"{name} blind to block swaps"
        assert dists[name][0] < dists[name][1], (
            f"{name} should degrade more gracefully under a block swap than "
            f"under a within-block reorder: {dists[name][0]} vs {dists[name][1]}"
        )

    for name in ("s3", "s6"):
        assert dists[name][0] > 0 and dists[name][1] > 0, (
            f"{name} should be sensitive to both reorderings"
        )


# --------------------------------------------------------------------------- #
# Range and degenerate corpora                                                 #
# --------------------------------------------------------------------------- #
def test_suite_keys_and_range():
    rng = np.random.RandomState(6)
    vocab = list(range(FIRST_SYMBOL, FIRST_SYMBOL + 14))
    seqs = [list(rng.choice(vocab, size=LENGTH)) for _ in range(30)]
    embs = [rng.randn(LENGTH, 16) for _ in seqs]
    concepts = rng.randn(30, 24)

    suite = E.topsim_suite(seqs, embs, concepts)
    assert set(suite) == set(E.TOPSIM_KEYS)
    for key, value in suite.items():
        assert np.isnan(value) or (-1.0 <= value <= 1.0), f"{key} = {value}"


def test_degenerate_corpus_is_nan_not_an_exception():
    rng = np.random.RandomState(7)

    # Constant signal side: every utterance identical.
    seqs = [[4, 5, 6, 7, 8] for _ in range(8)]
    embs = [rng.randn(LENGTH, 16) for _ in seqs]
    suite = E.topsim_suite(seqs, embs, rng.randn(8, 24))
    for key, value in suite.items():
        assert np.isnan(value), f"{key} = {value} on a constant-signal corpus"

    # Constant meaning side: every concept identical.
    seqs = [list(rng.randint(4, 18, size=LENGTH)) for _ in range(8)]
    embs = [rng.randn(LENGTH, 16) for _ in seqs]
    concepts = np.tile(rng.randn(1, 24), (8, 1))
    suite = E.topsim_suite(seqs, embs, concepts)
    for key, value in suite.items():
        assert np.isnan(value), f"{key} = {value} on a constant-meaning corpus"


# --------------------------------------------------------------------------- #
# Toy languages: does the suite measure what the chapter says it measures?     #
# --------------------------------------------------------------------------- #
N_ATTRIBUTES = LENGTH
N_VALUES = 3
N_CONCEPTS = 60


def _toy_meanings(rng):
    """``N_CONCEPTS`` distinct attribute tuples and their one-hot concepts."""
    seen, meanings = set(), []
    while len(meanings) < N_CONCEPTS:
        tup = tuple(rng.randint(0, N_VALUES, size=N_ATTRIBUTES))
        if tup not in seen:
            seen.add(tup)
            meanings.append(tup)
    concepts = np.zeros((N_CONCEPTS, N_ATTRIBUTES * N_VALUES))
    for i, tup in enumerate(meanings):
        for attribute, value in enumerate(tup):
            concepts[i, attribute * N_VALUES + value] = 1.0
    # Cosine distance between two of these is 1 - (matching attributes) / 5, so
    # meaning distance is exactly the number of differing attributes, rescaled.
    return meanings, concepts


def _toy_language(rng, free_order=False, synonyms=False):
    """
    A perfectly compositional toy language: one symbol per attribute value.

    ``free_order`` shuffles each utterance's symbols; ``synonyms`` gives each
    attribute value two interchangeable symbols with near-identical embeddings.
    Symbols for different attributes are disjoint, so a strict-order utterance
    is a positional encoding of the meaning tuple.
    """
    meanings, concepts = _toy_meanings(rng)
    n_forms = 2 if synonyms else 1
    per_value = N_ATTRIBUTES * N_VALUES * n_forms
    vocab_size = FIRST_SYMBOL + per_value

    def symbol(attribute, value, form):
        return FIRST_SYMBOL + (attribute * N_VALUES + value) * n_forms + form

    # One random unit embedding per attribute value; synonymous forms sit almost
    # on top of each other (cos ~ 0.999), distinct values are near-orthogonal.
    dim = 64
    table = np.zeros((vocab_size, dim))
    for attribute in range(N_ATTRIBUTES):
        for value in range(N_VALUES):
            anchor = rng.randn(dim)
            for form in range(n_forms):
                vector = anchor + 0.03 * rng.randn(dim)
                table[symbol(attribute, value, form)] = (
                    vector / np.linalg.norm(vector)
                )

    seqs = []
    for tup in meanings:
        message = [
            symbol(attribute, value, rng.randint(n_forms))
            for attribute, value in enumerate(tup)
        ]
        if free_order:
            rng.shuffle(message)
        seqs.append(message)

    return seqs, _table_embeddings(seqs, table), concepts


def test_toy_strict_order_asynonymous_scores_high_everywhere():
    """
    A perfectly compositional strict-order, synonym-free language scores high on
    all six -- such a language is in every signal set.

    S5 is the exception to "high means ~1": it lands around 0.73 rather than 1.0
    even here, and that is structural rather than a defect. A 5-symbol message
    has only 4 two-grams, so changing the attribute at an end position disturbs
    one two-gram while changing a middle one disturbs two. The hard 2-gram
    distance is therefore not a linear function of the number of differing
    attributes. S2 does not suffer as badly because its continuous ground cost
    gives partial credit to a two-gram that shares one constituent.
    """
    rng = np.random.RandomState(8)
    suite = E.topsim_suite(*_toy_language(rng))
    for key, value in suite.items():
        assert value > 0.7, f"{key} = {value:.3f} on a strict-order toy language"
    for key in ("topsim_s1", "topsim_s2", "topsim_s3", "topsim_s4", "topsim_s6"):
        assert suite[key] > 0.9, f"{key} = {suite[key]:.3f}"


def test_toy_free_order_keeps_s1_s4_and_loses_s3_s6():
    """Shuffling each utterance costs the order-sensitive sets, not the bags."""
    rng = np.random.RandomState(9)
    suite = E.topsim_suite(*_toy_language(rng, free_order=True))
    for key in ("topsim_s1", "topsim_s4"):
        assert suite[key] > 0.9, f"{key} = {suite[key]:.3f} should survive free order"
    for key in ("topsim_s3", "topsim_s6"):
        assert suite[key] < 0.6, f"{key} = {suite[key]:.3f} should collapse"
        assert suite[key] < suite["topsim_s1"] - 0.3, (
            f"{key} = {suite[key]:.3f} vs s1 = {suite['topsim_s1']:.3f}"
        )


def test_toy_synonymy_keeps_soft_and_loses_hard():
    """
    Interchangeable symbols with near-identical embeddings cost the hard
    variants and not the soft ones -- the whole point of parameterising the
    signal distance on the sender's own symbol embeddings.
    """
    rng = np.random.RandomState(10)
    suite = E.topsim_suite(*_toy_language(rng, synonyms=True))
    for key in ("topsim_s1", "topsim_s2", "topsim_s3"):
        assert suite[key] > 0.9, f"{key} = {suite[key]:.3f} should tolerate synonymy"
    for soft, hard in (("s1", "s4"), ("s2", "s5"), ("s3", "s6")):
        soft_value = suite[f"topsim_{soft}"]
        hard_value = suite[f"topsim_{hard}"]
        assert hard_value < 0.7, f"topsim_{hard} = {hard_value:.3f} should be degraded"
        assert hard_value < soft_value - 0.3, (
            f"topsim_{hard} = {hard_value:.3f} vs topsim_{soft} = {soft_value:.3f}"
        )


# --------------------------------------------------------------------------- #
# The `_static` leakage control and the ground-truth meaning space              #
# --------------------------------------------------------------------------- #
def _leaky_corpus(rng, n=120, d_concept=64, d_model=48):
    """
    A provably holistic language whose embeddings encode only the concept.

    Tokens are drawn independently of the meaning, so nothing about a message
    says anything about what it refers to and every variant should read 0. The
    embeddings, though, are a smooth deterministic function of the concept
    vector alone -- the worst case of what ``SenderGRULM`` does when it seeds
    its hidden state from ``init_h(concept)``.
    """
    concepts = rng.randn(n, d_concept)
    hidden = np.tanh(concepts @ (rng.randn(d_concept, d_model) / np.sqrt(d_concept)))
    embs = np.stack(
        [np.tanh(hidden @ (rng.randn(d_model, d_model) / np.sqrt(d_model)))
         for _ in range(LENGTH)],
        axis=1,
    )
    seqs = rng.randint(FIRST_SYMBOL, FIRST_SYMBOL + 14, size=(n, LENGTH)).tolist()
    return seqs, list(embs), concepts


def test_soft_variants_leak_the_meaning_space_without_the_control():
    """
    The failure `_static` exists to catch: on a language that says nothing, the
    soft variants still report a correlation, because their signal distance is
    a function of embeddings that are themselves a function of the concept.
    """
    seqs, embs, concepts = _leaky_corpus(np.random.RandomState(11))
    suite = E.topsim_suite(seqs, embs, concepts)

    for key in ("topsim_s2", "topsim_s3"):
        assert suite[key] > 0.2, (
            f"{key} = {suite[key]:.3f}; expected the uncontrolled soft variant "
            f"to report leakage on a holistic language"
        )
    for key in ("topsim_s4", "topsim_s5", "topsim_s6"):
        assert abs(suite[key]) < 0.1, (
            f"{key} = {suite[key]:.3f}; hard variants see only token identities "
            f"and cannot leak"
        )


def test_static_control_removes_the_leak():
    """Decontextualising the embeddings takes the soft variants back to zero."""
    seqs, embs, concepts = _leaky_corpus(np.random.RandomState(11))
    report = E.topsim_report(seqs, embs, concepts)

    for s in E.SOFT_SIGNAL_SETS:
        assert abs(report[f"topsim_{s}_static"]) < 0.1, (
            f"topsim_{s}_static = {report[f'topsim_{s}_static']:.3f} should be "
            f"~0 on a holistic language"
        )


def test_static_control_leaves_a_real_language_alone():
    """
    The control must cost nothing when there is nothing to control for. These
    toy embeddings are already a context-free lookup table, so decontextualising
    is the identity and `_static` must reproduce the raw value exactly.

    This is the property a permutation null does not have: permuting messages
    between concepts while each utterance keeps its embeddings leaves the
    embeddings encoding their original tokens, which in a compositional language
    still track the meaning, so such a control absorbs the real signal.
    """
    rng = np.random.RandomState(8)
    seqs, embs, concepts = _toy_language(rng, synonyms=True)
    report = E.topsim_report(seqs, embs, concepts)

    for s in E.SOFT_SIGNAL_SETS:
        assert np.isclose(report[f"topsim_{s}"], report[f"topsim_{s}_static"]), (
            f"topsim_{s} = {report[f'topsim_{s}']:.3f} but _static = "
            f"{report[f'topsim_{s}_static']:.3f}"
        )
        assert report[f"topsim_{s}_static"] > 0.9


def test_decontextualised_embeddings_are_constant_per_token():
    seqs = [[4, 5, 6, 4, 5], [5, 4, 7, 7, 6], [6, 6, 4, 5, 7]]
    rng = np.random.RandomState(12)
    embs = [rng.randn(LENGTH, 8) for _ in seqs]
    static = E.decontextualised_embeddings(seqs, embs)

    seen = {}
    for seq, table in zip(seqs, static):
        for position, token in enumerate(seq):
            if token in seen:
                assert np.allclose(seen[token], table[position]), (
                    f"token {token} has two different static embeddings"
                )
            seen[token] = table[position]

    # And the value is the mean over every occurrence.
    occurrences = [
        emb[position]
        for seq, emb in zip(seqs, embs)
        for position, token in enumerate(seq)
        if token == 4
    ]
    assert np.allclose(seen[4], np.mean(occurrences, axis=0))


def test_ground_truth_meaning_space_sees_a_collapsed_sender():
    """
    A sender that has collapsed onto one visual feature scores well against its
    own concept vectors -- it faithfully encodes what it represents -- and worse
    against the ground-truth logical forms, which know about the feature it
    dropped. Catching that gap is the point of the `topsim_gt_` family.
    """
    rng = np.random.RandomState(13)
    n_colours, n_shapes, dim = 6, 5, 32
    colour_vectors = rng.randn(n_colours, dim)

    # One symbol per colour, embedded by a fixed projection of the colour it
    # names, as a trained sender's symbol geometry would be. The soft variants
    # need the embedding space to mirror the meaning space for *some* reason;
    # positioning symbols by what they are used for is the honest one.
    symbol_table = np.zeros((FIRST_SYMBOL + n_colours, 16))
    symbol_table[FIRST_SYMBOL:] = colour_vectors @ rng.randn(dim, 16)

    formulas, concepts, seqs, embs = [], [], [], []
    for colour in range(n_colours):
        for shape in range(n_shapes):
            formulas.append(f"and c{colour} s{shape}")
            # The sender's own representation has lost the shape entirely: two
            # concepts of the same colour are the same point in its space.
            concepts.append(colour_vectors[colour])
            message = [FIRST_SYMBOL + colour] * LENGTH
            seqs.append(message)
            embs.append(symbol_table[message] + 0.01 * rng.randn(LENGTH, 16))

    report = E.topsim_report(seqs, embs, np.stack(concepts), formulas=formulas)
    assert set(report) == set(E.REPORT_KEYS)

    for s in E.SOFT_SIGNAL_SETS:
        own = report[f"topsim_{s}"]
        ground_truth = report[f"topsim_gt_{s}"]
        assert own > 0.5, f"topsim_{s} = {own:.3f} should be high on the sender's own space"
        assert ground_truth < own - 0.15, (
            f"topsim_gt_{s} = {ground_truth:.3f} should trail topsim_{s} = {own:.3f}"
        )
        # Nothing here is leaked: the embeddings are a function of the token.
        assert np.isclose(report[f"topsim_{s}_static"], own, atol=0.01)

    # The hard variants cannot tell the two meaning spaces apart here -- their
    # signal distance is binary (same colour or not), which says as much about
    # one space as the other. It is the soft variants that separate them.
    for s in ("s4", "s5", "s6"):
        assert abs(report[f"topsim_{s}"] - report[f"topsim_gt_{s}"]) < 0.05


def test_formula_distance_is_word_level_not_character_level():
    """
    Two formulas differing in one word are at distance 1, however many
    characters that word contains -- this is the paper's concept Edit distance.
    """
    from scipy.spatial.distance import squareform

    formulas = ["and red triangle", "and red circle", "or blue triangle"]
    full = squareform(E.formula_distance_condensed(formulas))
    assert full[0, 1] == 1.0, "one differing word is one edit"
    assert full[0, 2] == 2.0, "two differing words are two edits"
    assert full[0, 0] == 0.0


def test_report_omits_ground_truth_keys_without_formulas():
    rng = np.random.RandomState(14)
    seqs, embs, concepts = _toy_language(rng)
    report = E.topsim_report(seqs, embs, concepts)
    assert not any(key.startswith("topsim_gt_") for key in report)
    assert set(report) == set(E.TOPSIM_KEYS) | {
        f"topsim_{s}_static" for s in E.SOFT_SIGNAL_SETS
    }


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
