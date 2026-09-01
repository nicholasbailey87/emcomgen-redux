"""
Tests for the silhouette augmentation in code/data/generic.py.

Runnable without pytest:  python tests/test_silhouette.py

The augmentation exists to break the colour-only local minimum the paper reports
for ShapeWorld (~83% accuracy, appendix A.1). Three properties have to hold for
it to do that, and each is checked here.

First, that it actually removes colour. Converting to grayscale does not: the
six ShapeWorld colours sit at six well-separated luma values (blue 29 through
white 255), so a grayscale image still carries colour as a scalar that a single
conv filter can threshold. The test asserts both halves of this -- that the
lumas are distinct, which is why grayscale alone would fail, and that after
repainting all six colours render to the identical tensor.

Second, that it keeps shape, which is the entire point of removing colour, and
that it keeps it as a *binary* region. Every pixel above half the image's peak
luma is promoted to the fill and every other pixel to zero, so the output is
two-valued and an anti-aliased edge does not survive as a ramp. That is a worse
description of the object than the coverage blend it replaced on 2026-09-01, and
a better one to hand a receiver: the intervention is training-time only, and
`diagnostics/silhouette_shape_probe.py` measured that shape learned off coverage
edges does not transfer to the clean images eval uses (0.483 against the
threshold's 0.560, chance 0.306). See `silhouette`'s docstring for the table.

The colour is `silhouette_fill`, a per-channel fraction of maximum, defaulting
to the palette's own mean object colour (149, 149, 106). Which colour it is is a
claim about the receiver's input BatchNorm rather than about the language -- see
DEFAULT.toml -- so the tests here pin that the fill is honoured and configurable
and leave that argument where it is stated. What the tests do own is that no
palette colour is the fill, or that colour is silently exempt from the transform,
and that `fill * 255` is a whole number of stored levels, which under the
threshold means only that the fill is a colour the store can represent exactly.

The threshold makes the erasure sharper than the blend could: a pixel at
coverage `k` has `luma == k * peak` whatever the object's colour, so the
threshold falls at the same coverage for all six and the output histogram is
`{0, fill}` for every one of them. `test_the_threshold_is_colour_invariant` sweeps all 256 stored levels and finds
the single level where grey's coarser quantisation puts it on the other side.

Third, that the roll is per *game* and per *agent*. Per-game matters because
rolling per image would leave roughly (1-p) x 10 of a set's targets coloured and
the colour cue recoverable from the set. Per-agent matters because the receiver
is the side we mean to constrain: silhouetting the sender would teach it
shape-from-silhouette rather than the shape-from-colour-image competence eval
requires, so `silhouette_p_sender = 0` has to leave the sender's view untouched.
"""


import numpy as np
import torch

import _bootstrap  # noqa: F401

import parse_config
from data.generic import (
    DEFAULT_SILHOUETTE_FILL,
    ConceptDataset,
    silhouette,
)

# What a fully covered pixel comes back at, per dtype and per channel.
FILL_U8 = tuple(round(f * 255) for f in DEFAULT_SILHOUETTE_FILL)
# Through float32, so the equality assertions below are exact: the transform
#     multiplies a mask of exactly 1.0 by the float32 cast of this literal.
FILL_F32 = tuple(
    torch.tensor(DEFAULT_SILHOUETTE_FILL, dtype=torch.float32).tolist()
)

# The six ShapeWorld colours, as rendered.
COLOURS = np.array(
    [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 255, 255], [128, 128, 128]],
    dtype=np.uint8,
)
LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

# Every palette colour that is not itself the fill, which since 2026-09-01 is
# all six. `_is_silhouetted` calls a view silhouetted when every lit pixel sits
# at the fill, so a palette colour *equal* to the fill would be read as a
# repaint of itself and the dataset-level tests below would score it wrong. Two
# colours held this slot in turn: `white` under the white fill until 2026-08-29,
# then `gray` under the flat 0.5, which is exactly 128. A chromatic fill has no
# fixed point at all, so nothing is excluded and the dataset tests get their
# grey coverage back. `test_the_fill_collides_with_no_palette_colour` pins that;
# the name stays as it is because the exclusion is what it exists to express.
CHROMATIC = np.array(
    [c for c in COLOURS if not (c == np.array(FILL_U8)).all()], dtype=np.uint8
)

# The five colours whose maximum channel is 255, i.e. everything but `gray`.
#     Grey stores an edge in 129 levels where these use 256, which under the
#     coverage blend made it disagree with them everywhere and now makes it
#     disagree at exactly one. See `test_the_threshold_is_colour_invariant`.
BRIGHT = np.array([c for c in COLOURS if c.max() == 255], dtype=np.uint8)

N_GAMES, N_IMG, SIZE = 120, 40, 64


def _levels(out):
    """
    The distinct values present, per channel.

    Per channel rather than pooled: `sorted(out.unique())` reads the same for
        (149, 149, 106) and for a fill whose channels have been swapped, and
        the whole point of the chromatic fill is that the channels differ.
    """
    return [sorted(out[:, c].unique().tolist()) for c in range(3)]


def _square(colour, lo=20, hi=44):
    img = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    img[:, lo:hi, lo:hi] = np.asarray(colour, dtype=np.uint8).reshape(3, 1, 1)
    return img


def _disc(colour, r=12):
    img = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    mask = (yy - SIZE // 2) ** 2 + (xx - SIZE // 2) ** 2 <= r * r
    img[:, mask] = np.asarray(colour, dtype=np.uint8).reshape(3, 1)
    return img


def test_grayscale_would_leak_colour():
    """The premise: the six colours are separable by luma alone."""
    lumas = [
        int((torch.from_numpy(_square(c)).float() * LUMA).sum(0).max())
        for c in COLOURS
    ]
    assert len(set(lumas)) == len(COLOURS), lumas


def test_silhouette_erases_colour():
    batch = torch.from_numpy(np.stack([_square(c) for c in COLOURS]))
    out = silhouette(batch)
    for i in range(1, len(out)):
        assert torch.equal(out[0], out[i]), f"colour {i} survived silhouetting"
    assert _levels(out) == [[0, f] for f in FILL_U8]


def test_silhouette_keeps_shape():
    both = torch.from_numpy(np.stack([_square(COLOURS[0]), _disc(COLOURS[0])]))
    out = silhouette(both)
    assert not torch.equal(out[0], out[1])


def test_silhouette_preserves_dtype_and_range():
    u8 = torch.from_numpy(np.stack([_square(c) for c in COLOURS]))
    assert silhouette(u8).dtype == torch.uint8
    assert _levels(silhouette(u8)) == [[0, f] for f in FILL_U8]

    f32 = u8.float() / 255.0
    assert silhouette(f32).dtype == torch.float32
    assert _levels(silhouette(f32)) == [[0.0, f] for f in FILL_F32]


def test_silhouette_leaves_input_untouched():
    """The store is held in memory and `self.x[i]` is a view onto it."""
    batch = torch.from_numpy(np.stack([_square(c) for c in COLOURS]))
    before = batch.clone()
    silhouette(batch)
    assert torch.equal(batch, before)


def test_the_edge_is_promoted_to_the_fill():
    """
    A partly lit pixel above the threshold comes back at the full fill.

    This is the property that came back on 2026-09-01, and it is the exact
        inverse of `test_coverage_is_preserved`, which pinned the blend that
        held this slot from e884662. The repainted region is therefore larger
        than the object it replaces and the per-channel image mean overshoots
        with it -- a real cost, accepted because shape read off a coverage edge
        does not survive the trip to the clean images eval uses.

    The threshold falls between stored levels 127 and 128, i.e. at a coverage
        of one half, and it falls there for every bright colour. See
        `test_the_threshold_is_colour_invariant`.
    """
    img = np.zeros((1, 3, SIZE, SIZE), dtype=np.uint8)
    # A solid blue block, with one row just over half intensity and one just
    #     under, standing in for rows the renderer covered either side of half.
    img[0, :, 20:40, 20:44] = np.array([0, 0, 255], dtype=np.uint8).reshape(3, 1, 1)
    img[0, :, 40, 20:44] = np.array([0, 0, 128], dtype=np.uint8).reshape(3, 1)
    img[0, :, 41, 20:44] = np.array([0, 0, 127], dtype=np.uint8).reshape(3, 1)

    out = silhouette(torch.from_numpy(img))

    assert out[0, :, 30, 30].tolist() == list(FILL_U8)
    assert out[0, :, 40, 30].tolist() == list(FILL_U8)
    assert int(out[0, :, 41, 30].max()) == 0
    assert int(out[0, :, 50, 30].max()) == 0
    # Two-valued, which is the whole of what an edge carries now.
    assert _levels(out) == [[0, f] for f in FILL_U8]


def _edge_pixel(colour, n):
    """
    One pixel stored at level `n` of an object of `colour`, silhouetted.

    The image also carries a fully covered pixel, because the transform
        normalises by the image's peak luma and would otherwise read the edge
        pixel itself as the whole object.
    """
    img = np.zeros((1, 3, 8, 8), dtype=np.uint8)
    c = np.asarray(colour, dtype=np.float64)
    img[0, :, 0, 0] = colour
    img[0, :, 1, 1] = np.round(c * n / 255)
    return tuple(silhouette(torch.from_numpy(img))[0, :, 1, 1].tolist())


def test_the_threshold_is_colour_invariant():
    """
    All six colours cross the threshold at the same coverage, bar one level.

    A pixel at coverage `k` on a black ground has `luma == k * peak` whatever
        the object's colour, so `luma > peak / 2` is a statement about `k` and
        nothing else, and the output is `{0, fill}` for every colour. That is
        the erasure the coverage blend could not make: under it, grey resolved
        an edge in 129 levels where the rest used 256 and its output skipped
        specific intensity values -- a structural gap a classifier found at
        ~0.97 recall against a chance of 0.167 (docs/dubious-claims.md).

    The exception, and it is the whole residual. The threshold is taken on the
        *stored* image, and grey stores coverage `k` as `round(128k)` where a
        bright colour stores `round(255k)`. Grey therefore turns on at
        `round(128k) >= 65`, i.e. `k > 0.5039`, and the rest at `k >= 0.502`.
        Exactly one of the 256 sampled levels falls in that gap, `n = 128`, and
        there grey is off while the other five are on.

    So: a boundary that can differ by a pixel, against a value histogram that
        differed everywhere. Asserting the divergence rather than fixing it, as
        `test_grey_resolves_coverage_more_coarsely` did before it, because the
        day the gap widens somebody should notice.
    """
    disagreed = []
    for n in range(256):
        got = {tuple(colour): _edge_pixel(colour, n) for colour in COLOURS}
        if len(set(got.values())) != 1:
            disagreed.append((n, got))

    assert [n for n, _ in disagreed] == [128], disagreed

    n, got = disagreed[0]
    assert got[(128, 128, 128)] == (0, 0, 0)
    for colour in BRIGHT:
        assert got[tuple(colour)] == FILL_U8

    # And the five that quantise alike agree at every level, including 128.
    for n in range(256):
        got = {tuple(colour): _edge_pixel(colour, n) for colour in BRIGHT}
        assert len(set(got.values())) == 1, f"level {n} disagreed: {got}"


def test_the_fill_is_an_integer_number_of_levels():
    """
    `fill * 255` is a whole number of stored levels in every channel.

    This carried more weight under the coverage blend, where it was what kept
        anti-aliased edges off colour-dependent rounding ties: a stored edge
        pixel was `round(n * F / 255)`, and with `F = 127.5` every odd `n` was
        a tie broken on the last bits of a colour-dependent float32, which is
        how 43 of 256 levels came to leak. There are no anti-aliased edges to
        round now.

    What it still means is that the fill is a colour the store can represent
        exactly rather than one that lands between two levels, so an edit to a
        sloppy literal like 0.58 -- whose product is 147.9, i.e. neither 148
        nor anything with a reason behind it -- fails here with the reason
        attached. The tolerance is tight but not exact: the TOML literal is
        0.584313725, whose product is 149.000000875.
    """
    for f in DEFAULT_SILHOUETTE_FILL:
        assert abs(f * 255 - round(f * 255)) < 1e-4, f


def test_the_fill_collides_with_no_palette_colour():
    """
    No palette colour is the transform's fixed point.

    Under the flat 0.5 the fill was exactly 128, which is `gray`, so a grey
        object came back bit-identical -- one colour in six silently exempt
        from an intervention whose entire purpose is to remove colour. Under
        the white fill before it, `white` held the same slot.
    """
    assert len(CHROMATIC) == len(COLOURS)

    grey = torch.from_numpy(_square(np.array([128, 128, 128], dtype=np.uint8))[None])
    assert not torch.equal(silhouette(grey), grey)


def test_the_fill_is_actually_written_to_three_channels():
    """
    The interior of a covered object reads the fill, per channel.

    The output line used to end `.expand_as(imgs)`, which broadcast a single
        mask channel across three. That was correct while the fill was
        achromatic and is silently wrong now: it would write one channel three
        times, degrading the chromatic fill back to a grey and restoring the
        `gray` collision, and nothing else here would fail. Only an interior
        value catches it.
    """
    out = silhouette(torch.from_numpy(_square(COLOURS[0])[None]))
    assert out[0, :, 32, 32].tolist() == [149, 149, 106]
    assert out[0, :, 32, 32].tolist() == list(FILL_U8)
    assert int(out[0, :, 0, 0].max()) == 0


def test_the_fill_is_honoured():
    """
    `silhouette_fill` reaches the pixels, so the config key is not inert.

    Asserted at a value the default is not, because the default is what every
        other test here already pins.
    """
    out = silhouette(torch.from_numpy(_square(CHROMATIC[0])[None]), fill=0.25)
    assert _levels(out) == [[0, round(0.25 * 255)]] * 3

    f32 = torch.from_numpy(_square(CHROMATIC[0])[None]).float() / 255.0
    assert _levels(silhouette(f32, fill=0.25)) == [[0.0, 0.25]] * 3

    # A triple, which is what the config now sends and what `expand_as` used to
    #     flatten back to one channel without failing anything.
    out = silhouette(
        torch.from_numpy(_square(CHROMATIC[0])[None]), fill=(0.2, 0.4, 0.6)
    )
    assert _levels(out) == [[0, 51], [0, 102], [0, 153]]


def test_the_fill_reaches_the_dataset():
    """
    `ConceptDataset` forwards its `silhouette_fill` rather than dropping it.

    The transform tests above call `silhouette` directly, so nothing there
        would notice the kwarg going missing between the config and the call.
    """
    np.random.seed(0)
    view = _dataset(0.0, 1.0, fill=0.25)[0][2]
    assert _levels(view) == [[0, round(0.25 * 255)]] * 3

    np.random.seed(0)
    view = _dataset(0.0, 1.0, fill=(0.2, 0.4, 0.6))[0][2]
    assert _levels(view) == [[0, 51], [0, 102], [0, 153]]


def _rejected_by_the_config(fill):
    """`validate_config` over the defaults with `silhouette_fill` overridden."""
    config = parse_config.get_config()
    config['data']['silhouette_fill'] = fill
    try:
        parse_config.validate_config(config)
    except parse_config.InvalidConfig:
        return True
    return False


def test_the_config_rejects_a_fill_that_is_not_a_colour():
    """
    Rejected at parse time, with the key named, rather than at the first batch.

    `silhouette_fill` stopped being a scalar on 2026-09-01, and the check it
        used to share with `silhouette_p_*` would have raised `TypeError` on
        the list -- an unreadable traceback out of a comparison, from every run
        and most of the suite at once.
    """
    assert _rejected_by_the_config([0.5, 0.5])
    assert _rejected_by_the_config([0.5, 0.5, 0.5, 0.5])
    assert _rejected_by_the_config([-0.1, 0.5, 0.5])
    assert _rejected_by_the_config([0.5, 0.5, 1.1])
    assert _rejected_by_the_config(1.5)
    assert _rejected_by_the_config("0.5")
    assert _rejected_by_the_config(True)


def test_the_config_accepts_both_a_scalar_and_a_triple():
    assert not _rejected_by_the_config(0.5)
    assert not _rejected_by_the_config([0.584313725, 0.584313725, 0.415686275])
    assert not _rejected_by_the_config(list(DEFAULT_SILHOUETTE_FILL))


def test_all_black_image_stays_black():
    """Guard on the relative threshold: peak luma of zero must not invert."""
    black = torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8)
    assert int(silhouette(black).max()) == 0


def _dataset(p_sender, p_receiver, seed=0, fill=DEFAULT_SILHOUETTE_FILL):
    rng = np.random.default_rng(seed)
    x = np.zeros((N_GAMES, N_IMG, 3, SIZE, SIZE), dtype=np.uint8)
    for g in range(N_GAMES):
        for i in range(N_IMG):
            x[g, i] = _square(CHROMATIC[rng.integers(len(CHROMATIC))])

    labels = np.zeros((N_GAMES, N_IMG), dtype=bool)
    labels[:, : N_IMG // 2] = True

    data = {
        "x": x,
        "labels": labels,
        "langs": np.array([["red"] for _ in range(N_GAMES)], dtype=object),
        "metadata": np.zeros(N_GAMES, dtype=int),
    }
    vocab = {
        "w2i": {"<PAD>": 0, "<s>": 1, "</s>": 2, "<UNK>": 3, "red": 4},
        "i2w": {0: "<PAD>", 1: "<s>", 2: "</s>", 3: "<UNK>", 4: "red"},
    }
    return ConceptDataset(
        data,
        vocab,
        n_examples=20,
        augment=True,
        silhouette_p_sender=p_sender,
        silhouette_p_receiver=p_receiver,
        silhouette_fill=fill,
    )


def _is_silhouetted(view, fill=FILL_U8):
    """
    A view is silhouetted iff every lit pixel sits at exactly `fill`.

    Not "achromatic and at the fill" any more -- the fill is a colour, so the
        criterion is a per-channel match against the triple. This is what
        `test_rates_are_*` measure, so a wrong answer here makes those tests
        lie rather than fail, which is worth the caution.

    `fill` is a parameter because `_dataset` takes one: hardcoding the default
        here while the dataset was built at some other fill would have read
        every game as un-silhouetted and passed `test_rates_are_off_when_zero`
        for the wrong reason.

    The games these read are built from `_square` and `_disc`, both of which
        mask hard. Under the threshold every repainted view holds only the fill
        and zero anyway, whatever it was rendered from -- but this was written
        against the coverage blend, where a soft edge would have read as a view
        that was not silhouetted, and the geometry is the reason it did not.
    """
    vals = view.reshape(view.shape[0], 3, -1)
    target = torch.tensor(fill, dtype=vals.dtype).reshape(1, 3, 1)
    lit = vals.amax(1) > 0
    return bool((vals == target).all(1)[lit].all())


def _rates(p_sender, p_receiver, seed=0):
    np.random.seed(seed)
    ds = _dataset(p_sender, p_receiver)
    sender = [_is_silhouetted(ds[i][0]) for i in range(N_GAMES)]
    np.random.seed(seed)
    ds = _dataset(p_sender, p_receiver)
    receiver = [_is_silhouetted(ds[i][2]) for i in range(N_GAMES)]
    return np.mean(sender), np.mean(receiver)


def test_rates_are_off_when_zero():
    assert _rates(0.0, 0.0) == (0.0, 0.0)


def test_rates_are_on_when_one():
    assert _rates(1.0, 1.0) == (1.0, 1.0)


def test_receiver_only_leaves_sender_untouched():
    """The default regime: `silhouette_p_sender = 0`, receiver at p."""
    sender_rate, receiver_rate = _rates(0.0, 1.0)
    assert sender_rate == 0.0
    assert receiver_rate == 1.0


def test_rates_track_p():
    sender_rate, receiver_rate = _rates(0.0, 0.5)
    # ~4 sigma at n = 120.
    assert abs(receiver_rate - 0.5) < 0.18
    assert sender_rate == 0.0


def test_roll_is_per_game_not_per_image():
    """Every image in a silhouetted view must be silhouetted, or none."""
    np.random.seed(0)
    ds = _dataset(0.0, 0.5)
    mixed = 0
    for i in range(N_GAMES):
        view = ds[i][2]
        per_image = [_is_silhouetted(view[j : j + 1]) for j in range(view.shape[0])]
        if any(per_image) and not all(per_image):
            mixed += 1
    assert mixed == 0, f"{mixed} games mixed silhouetted and coloured images"


def test_shapes_and_dtypes_unchanged():
    np.random.seed(0)
    plain = _dataset(0.0, 0.0)[0]
    np.random.seed(0)
    silhouetted = _dataset(0.0, 1.0)[0]
    assert len(plain) == len(silhouetted)
    for a, b in zip(plain[:4], silhouetted[:4]):
        assert a.shape == b.shape and a.dtype == b.dtype


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
