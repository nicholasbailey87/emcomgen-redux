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
that it keeps it at the *coverage* the object had rather than binarised. The
transform scales each pixel by `luma / peak`, so an anti-aliased edge survives
with its partial coverage intact; the threshold it replaced promoted every
partly-lit edge pixel to fully lit and grew the repainted region.

The intensity is `silhouette_fill` of maximum in all three channels, half by
default. That is a claim about the receiver's input BatchNorm rather than about
the language -- see DEFAULT.toml -- so the tests here pin that the fill is
honoured and configurable, and leave the argument where it is stated.

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

from data.generic import (
    DEFAULT_SILHOUETTE_FILL,
    ConceptDataset,
    silhouette,
)

# What a fully covered pixel comes back at, per dtype.
FILL_U8 = round(DEFAULT_SILHOUETTE_FILL * 255)
FILL_F32 = DEFAULT_SILHOUETTE_FILL

# The six ShapeWorld colours, as rendered.
COLOURS = np.array(
    [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 255, 255], [128, 128, 128]],
    dtype=np.uint8,
)
LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

# `_is_silhouetted` calls a view silhouetted when every lit pixel sits at the
# fill, so a genuinely *gray* object is indistinguishable from a repaint of one:
# `gray` is 128 and the default fill is 0.5 of 255. That is harmless for the
# transform tests, which know what they fed in, but it would make the
# dataset-level tests below read gray squares as silhouettes. Those build their
# games from the other five colours instead.
#
# White held this slot until 2026-08-29, for exactly the same reason against the
# old white fill. One palette colour is always the transform's fixed point; the
# fill decides which.
CHROMATIC = np.array(
    [c for c in COLOURS if not (c == FILL_U8).all()], dtype=np.uint8
)

N_GAMES, N_IMG, SIZE = 120, 40, 64


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
    assert sorted(out.unique().tolist()) == [0, FILL_U8]


def test_silhouette_keeps_shape():
    both = torch.from_numpy(np.stack([_square(COLOURS[0]), _disc(COLOURS[0])]))
    out = silhouette(both)
    assert not torch.equal(out[0], out[1])


def test_silhouette_preserves_dtype_and_range():
    u8 = torch.from_numpy(np.stack([_square(c) for c in COLOURS]))
    assert silhouette(u8).dtype == torch.uint8
    assert sorted(silhouette(u8).unique().tolist()) == [0, FILL_U8]

    f32 = u8.float() / 255.0
    assert silhouette(f32).dtype == torch.float32
    assert sorted(silhouette(f32).unique().tolist()) == [0.0, FILL_F32]


def test_silhouette_leaves_input_untouched():
    """The store is held in memory and `self.x[i]` is a view onto it."""
    batch = torch.from_numpy(np.stack([_square(c) for c in COLOURS]))
    before = batch.clone()
    silhouette(batch)
    assert torch.equal(batch, before)


def test_coverage_is_preserved():
    """
    A half-covered pixel comes back at half the fill, not at the fill.

    This is the property the `luma > peak / 2` threshold destroyed: it promoted
        every partly-lit edge pixel to fully lit, so the repainted region was
        larger than the object it replaced and the per-channel image mean
        overshot by the difference. ShapeWorld renders anti-aliased edges, so
        this is every object's boundary rather than a corner case.
    """
    img = np.zeros((1, 3, SIZE, SIZE), dtype=np.uint8)
    # A solid blue block, with one row at half intensity standing in for a row
    #     of pixels the renderer covered halfway.
    img[0, :, 20:40, 20:44] = np.array([0, 0, 255], dtype=np.uint8).reshape(3, 1, 1)
    img[0, :, 40, 20:44] = np.array([0, 0, 128], dtype=np.uint8).reshape(3, 1)

    out = silhouette(torch.from_numpy(img))

    assert int(out[0, 0, 30, 30]) == FILL_U8
    assert int(out[0, 0, 40, 30]) == round(FILL_U8 * 128 / 255)
    assert int(out[0, 0, 50, 30]) == 0


def test_partial_coverage_is_colour_invariant():
    """
    Two colours at the same coverage repaint identically.

    The whole-object case is `test_silhouette_erases_colour`; this is the edge
        case, and it is the one that makes the coverage blend safe to swap in
        for the threshold. It holds because a pixel at coverage `k` on a black
        ground has `luma == k * peak` whatever the object's colour, so
        `luma / peak` is the coverage and nothing else.
    """
    def half_covered(colour):
        img = np.zeros((1, 3, SIZE, SIZE), dtype=np.uint8)
        c = np.asarray(colour, dtype=np.int64).reshape(3, 1, 1)
        img[0, :, 20:40, 20:44] = c
        img[0, :, 40, 20:44] = (c // 2).reshape(3, 1)
        return silhouette(torch.from_numpy(img))

    first = half_covered(CHROMATIC[0])
    for colour in CHROMATIC[1:]:
        assert torch.equal(first, half_covered(colour))


def test_the_fill_is_honoured():
    """
    `silhouette_fill` reaches the pixels, so the config key is not inert.

    Asserted at a value the default is not, because the default is what every
        other test here already pins.
    """
    out = silhouette(torch.from_numpy(_square(CHROMATIC[0])[None]), fill=0.25)
    assert sorted(out.unique().tolist()) == [0, round(0.25 * 255)]

    f32 = torch.from_numpy(_square(CHROMATIC[0])[None]).float() / 255.0
    assert sorted(silhouette(f32, fill=0.25).unique().tolist()) == [0.0, 0.25]


def test_the_fill_reaches_the_dataset():
    """
    `ConceptDataset` forwards its `silhouette_fill` rather than dropping it.

    The transform tests above call `silhouette` directly, so nothing there
        would notice the kwarg going missing between the config and the call.
    """
    np.random.seed(0)
    view = _dataset(0.0, 1.0, fill=0.25)[0][2]
    assert sorted(view.unique().tolist()) == [0, round(0.25 * 255)]


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


def _is_silhouetted(view):
    """
    A view is silhouetted iff every lit pixel is achromatic and at the fill.

    The games these read are built from `_square` and `_disc`, both of which
        mask hard, so a repainted view is exactly {0, FILL_U8} and there are no
        partial-coverage pixels to allow for.
    """
    vals = view.reshape(view.shape[0], 3, -1)
    lit = vals.amax(1) > 0
    return bool(
        ((vals == FILL_U8) | (vals == 0)).all()
        and (vals.amin(1)[lit] == FILL_U8).all()
    )


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
