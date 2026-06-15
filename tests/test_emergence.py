"""
Tests for the vectorized topsim metrics in code/emergence.py.

Runnable without pytest:  python tests/test_emergence.py
The headline check asserts the new rapidfuzz-based Levenshtein topsim is
numerically identical to the old editdistance + pure-Python pdist it replaces.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import emergence as E  # noqa: E402


# --- reference: the OLD implementation this change replaces ----------------- #
def _old_topsim(meanings, messages):
    import editdistance
    from scipy.spatial import distance
    from scipy.stats import spearmanr

    def old_edit(x, y):
        return editdistance.eval(x, y) / ((len(x) + len(y)) / 2)

    def old_pdist(X, metric):
        m = len(X)
        dm = np.empty((m * (m - 1)) // 2)
        k = 0
        for i in range(m - 1):
            for j in range(i + 1, m):
                dm[k] = metric(X[i], X[j])
                k += 1
        return dm

    md = distance.pdist(meanings, "hamming")
    sd = old_pdist(messages, old_edit)
    return spearmanr(md, sd, nan_policy="raise").correlation


def _synthetic(seed=0, n=80, vocab=12, attr_dim=8):
    rng = np.random.RandomState(seed)
    meanings = rng.randint(0, 2, size=(n, attr_dim)).astype(float)
    msgs = [list(rng.randint(0, vocab, size=rng.randint(1, 5))) for _ in range(n)]
    reprs = rng.randn(n, 16)
    return meanings, msgs, reprs, vocab


def test_levenshtein_topsim_matches_old():
    meanings, msgs, _, _ = _synthetic()
    mean_cond = E.vector_condensed(meanings, metric="hamming")
    new = E.spearman_topsim(mean_cond, E.levenshtein_condensed(msgs))
    old = _old_topsim(meanings, msgs)
    assert abs(new - old) < 1e-9, f"lev topsim mismatch: new={new} old={old}"


def test_grid_in_range_and_keys():
    meanings, msgs, reprs, V = _synthetic()
    mean_cond = E.vector_condensed(meanings, metric="hamming")
    grid, pc = E.topsim_grid(msgs, reprs, vocab_size=V, concept_meaning_cond=mean_cond)
    assert set(grid) == {
        "ts_lev", "ts_tanimoto", "ts_sif",
        "reprts_lev", "reprts_tanimoto", "reprts_sif",
    }
    for k, v in grid.items():
        assert np.isnan(v) or (-1.0 <= v <= 1.0), f"{k}={v} out of range"
    assert pc is not None


def test_pc_reuse_on_prototype_subset():
    _, msgs, reprs, V = _synthetic()
    _, pc = E.sif_condensed(msgs, V, return_pc=True)
    sif_small, _ = E.sif_condensed(msgs[:20], V, pc=pc, return_pc=True)
    assert sif_small.shape[0] == 20 * 19 // 2


def test_low_rank_guard_does_not_crash():
    # Degenerate corpus: a single token repeated -> co-occurrence near rank 0.
    deg = [[0, 0, 0] for _ in range(10)]
    sif, pc = E.sif_condensed(deg, vocab_size=12, return_pc=True)
    assert np.isnan(sif).all() and pc is None
    grid, _ = E.topsim_grid(
        deg, np.random.randn(10, 16), vocab_size=12,
        concept_meaning_cond=E.vector_condensed(np.random.rand(10, 8), "hamming"),
    )
    assert np.isnan(grid["ts_sif"])  # degenerate SIF -> NaN, no exception


def test_tanimoto_empty_messages():
    # Two empty messages are identical (0); empty vs non-empty is maximal (1).
    cond = E.tanimoto_condensed([[], [1], [2, 3], []], vocab_size=12)
    # pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    assert cond[2] == 0.0  # both empty
    assert cond[0] == 1.0  # empty vs [1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
