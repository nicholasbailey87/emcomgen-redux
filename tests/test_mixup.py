"""
Tests for `[data] mixup_alpha` in code/data/generic.py.

Runnable without pytest:  python tests/test_mixup.py

Each of the listener's candidates is replaced by a blend of itself and another
drawn from the same set with replacement, and its label by the same blend --
Zhang et al. 2018 (arXiv:1710.09412), unchanged. Positives and negatives mix
together, so the target becomes continuous.

The label states how the image was *built*, not what it depicts, and that
distinction is why there is no test here asserting anything about the picture's
meaning. A ShapeWorld world holds one object, so a blend of a red circle and a
blue square is two half-lit ghosts rather than an object that is 70% red. What
the tests below pin is that the construction is the one claimed: that image and
label are blended by the *same* weight, that nothing leaves [0, 1], that the
store the caller handed in is not written through, and that the whole thing is
off and inert by default.

`test_the_image_and_the_label_share_a_weight` is the one that matters. It works
by making each candidate a constant image whose value is proportional to its own
label, so that after blending the image's value has to stay proportional to the
blended label -- for every candidate, under weights the test never observes
directly. A transform that drew two independent weights, or blended the labels
against a different partner than the images, fails it.
"""

import inspect

import numpy as np
import torch

import _bootstrap  # noqa: F401

from data.generic import ConceptDataset


ALPHA = 1.0
LIT = 200.0


class _Probe(ConceptDataset):
    """The transform alone, without a store or a game behind it."""

    def __init__(self, mixup_alpha=0.0):
        self.mixup_alpha = mixup_alpha


def _candidates(dtype=torch.uint8):
    """Ten positives then ten negatives, as `split_spk_lis` builds them."""
    imgs = torch.zeros(20, 3, 8, 8, dtype=dtype)
    imgs[:, 0] = torch.arange(20).reshape(20, 1, 1).to(dtype) * 10
    labels = torch.zeros(20, dtype=torch.uint8)
    labels[:10] = 1
    return imgs, labels


def test_it_is_off_by_default():
    assert inspect.signature(ConceptDataset.__init__).parameters[
        "mixup_alpha"
    ].default == 0.0


def test_off_is_a_passthrough_for_images_and_labels():
    imgs, labels = _candidates()
    out, y = _Probe(0.0)._apply_mixup(imgs, labels)
    assert torch.equal(out, imgs)
    assert torch.equal(y, labels)


def test_the_image_and_the_label_share_a_weight():
    """
    See the module docstring. Each candidate is a constant image at `LIT` times
        its own label, so `image == LIT * label` before the blend; both sides
        are linear in the same weight, so it has to hold after it too.
    """
    imgs, labels = _candidates(dtype=torch.float32)
    imgs[:] = labels.reshape(20, 1, 1, 1).to(torch.float32) * LIT

    np.random.seed(0)
    out, y = _Probe(ALPHA)._apply_mixup(imgs, labels)

    assert torch.allclose(out, y.reshape(20, 1, 1, 1) * LIT, atol=1e-4)


def test_the_labels_go_continuous_and_stay_in_range():
    imgs, labels = _candidates()
    np.random.seed(0)
    _, y = _Probe(ALPHA)._apply_mixup(imgs, labels)

    assert y.dtype == torch.float32
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    # A blend of two candidates with the same label lands back on it, so only
    #     the cross-label pairs go soft -- roughly half of them at 10 and 10.
    assert ((y > 0.01) & (y < 0.99)).any(), "no soft labels at all"


def test_the_callers_tensors_are_not_written_through():
    """`self.x[i]` is a view onto the shared store when it is held in memory."""
    imgs, labels = _candidates()
    before_imgs, before_labels = imgs.clone(), labels.clone()
    np.random.seed(0)
    _Probe(ALPHA)._apply_mixup(imgs, labels)
    assert torch.equal(imgs, before_imgs)
    assert torch.equal(labels, before_labels)


def test_shape_and_image_dtype_survive():
    for dtype in (torch.uint8, torch.float32):
        imgs, labels = _candidates(dtype=dtype)
        np.random.seed(0)
        out, _ = _Probe(ALPHA)._apply_mixup(imgs, labels)
        assert out.shape == imgs.shape
        assert out.dtype == dtype


def test_an_integer_store_is_rounded_rather_than_truncated():
    """
    Truncation would bias every blend down, and on a black background down is
        towards the background.
    """
    source = inspect.getsource(ConceptDataset._apply_mixup)
    assert "blended.round()" in source


def test_it_is_listener_only_and_train_only():
    """
    The speaker describes clean images. Pinned by reading the call site, since
        `__getitem__` needs a whole dataset behind it to exercise directly.
    """
    # `util.return_index` wraps `__getitem__` in a closure and does not carry
    #     `functools.wraps`, so the attribute is the wrapper and its source is
    #     three lines of decorator. The wrapped function is the closed-over cell.
    wrapped = ConceptDataset.__getitem__.__closure__[0].cell_contents
    source = inspect.getsource(wrapped)
    call = "lis_inp, lis_label = self._apply_mixup(lis_inp, lis_label)"
    assert call in source
    assert "self._apply_mixup(spk_inp" not in source
    # Inside the `if self.augment:` block, which is what makes it train-only.
    assert source.index("if self.augment:") < source.index(call)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"{name} ok")
