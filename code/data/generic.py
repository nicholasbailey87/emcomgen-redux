import numpy as np
import torch
import torch.nn.functional as F
import os

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from . import language
from . import util


def vis_image(inp, overwrite=True, **kwargs):
    img_fname = f"{kwargs['name']}_{kwargs['epoch']}_{kwargs['split']}_{kwargs['game_i']}_{kwargs['i']}.jpg"
    img_f = os.path.join(kwargs["exp_dir"], "images", img_fname)
    img_html = f"""<img src="{os.path.join('images', img_fname)}">"""
    if os.path.exists(img_f) and not overwrite:
        return img_html
    inp = inp.permute((1, 2, 0)).numpy()
    if inp.dtype == np.float32:
        inp = np.round(inp * 255).astype(np.uint8)
    Image.fromarray(inp).save(img_f)
    return img_html


# Rec.601 luma weights, i.e. what a grayscale conversion would apply.
_LUMA = torch.tensor([0.299, 0.587, 0.114])


# ShapeWorld's own mean object colour, (149, 149, 106) of 255. Chromatic so that
#     no palette colour is a fixed point, and an integer number of levels in
#     every channel so that it is a colour the store can represent exactly; both
#     are explained under `silhouette` and in docs/data.md. This was a flat 0.5
#     until 2026-09-01, which collided with the palette's `gray` -- and, while
#     the transform blended by coverage, leaked colour through the edges too.
DEFAULT_SILHOUETTE_FILL = (149 / 255, 149 / 255, 106 / 255)


def _as_fill(fill, device):
    """
    `fill` as a (1, 3, 1, 1) float tensor, from a scalar or a length-3 sequence.

    A scalar is broadcast to all three channels. It is not the default any more
        but it stays supported: it is the natural thing to write in a one-off
        probe, and the tests use it to check the key is honoured at a value the
        default is not.
    """
    out = torch.as_tensor(fill, dtype=torch.float32, device=device).reshape(-1)
    if out.numel() == 1:
        out = out.expand(3)
    elif out.numel() != 3:
        raise ValueError(
            f"`fill` must be a scalar or a length-3 sequence, got {fill!r}"
        )
    # `reshape` rather than `view`: `expand` above leaves a stride-0 tensor.
    return out.reshape(1, 3, 1, 1)


def silhouette(imgs, fill=DEFAULT_SILHOUETTE_FILL):
    """
    Repaint each image's object in a flat `fill` colour, keeping its shape.

    `imgs` is (n, C, H, W) with a black background. Every pixel lit above half
    the image's peak luma comes back at `fill` of maximum intensity and every
    other pixel at zero, so the output is two-valued per channel. dtype and
    range are preserved -- `fill` is read against 255 for integer images and
    against 1.0 for floating-point ones. Returns a new tensor; the input is left
    untouched.

    `fill` is a length-3 per-channel fraction, or a scalar broadcast to three.

    A threshold rather than coverage, since 2026-09-01. Between e884662 and this
    version the transform scaled each pixel by its coverage `luma / peak`, which
    kept an anti-aliased edge at the coverage it had instead of promoting it to
    fully lit. That is the better description of the object, and it is the wrong
    thing to hand a receiver. The intervention is training-time only and eval is
    never silhouetted, so what matters is not how much of the object survives the
    repainting but how much of *it* survives the trip back to clean images -- and
    `diagnostics/silhouette_shape_probe.py` (cluster job 123354) measured a
    `Conv4` trained on each fill and tested on both:

        arm                in_domain    clean
        clean                  0.999    0.999
        white_threshold        0.794    0.560
        white_coverage         1.000    0.483
        fill_coverage          1.000    0.486     (chance 0.306)

    The coverage arms are perfectly readable under their own repainting and lose
    almost all of it at eval. The threshold is the only arm that gives up
    in-domain accuracy and keeps more of it on clean images.

    That table is one fit per arm, and it does not survive its own repeat.
    Re-run unchanged at the same seed on the same GPU, `white_threshold` read
    0.403 where it had read 0.560 (jobs 123583, 123354), because convolution
    backward on cuDNN accumulates with atomics; a third run read the three live
    fills at 0.412, 0.553 and 0.506. Six single-fit readings, all between 0.40
    and 0.56, none separable from another. What survives it: every repainting
    arm sits well above chance and far below `clean`, and the threshold costs
    in-domain readability on every run (0.79-0.97 against the blend's 1.000).
    Which fill transfers best is open. `diagnostics/silhouette_shape_probe.py`
    now fits each arm five times and reports a range; run it before quoting a
    number from here.


    Colour invariance, now exact. For a flat object on black a pixel at coverage
    `k` has `luma == k * peak` whatever the object's colour, so the threshold
    falls in the same place for all six palette colours and the output is
    `{0, fill}` for every one of them. The value histogram no longer depends on
    the colour at all, which retires the leak the coverage version could not fix:
    grey has HSV Value 0.502 and so resolved coverage in 129 steps where the rest
    used 256, skipping specific intensity values in a way a classifier found at
    ~0.97 recall against a chance of 0.167.

    What is left of it is a different kind of thing. The threshold is taken on
    the *stored* image, so a pixel whose true coverage lies within one
    quantisation step of 0.5 can fall either side of it, and grey's steps are
    twice as wide as everything else's. That moves a boundary by up to a pixel;
    it does not put a colour-dependent gap in every image's histogram.

    The cost, which e884662 was right about and which we are accepting: the
    repainted region is larger than the object it replaces, because every partly
    lit edge pixel is promoted to fully lit, and the per-channel image mean
    overshoots with it.

    Chromatic, and an integer number of levels. The default is the palette's own
    mean object colour, (149, 149, 106), and the argument for it is untouched by
    the edge treatment. The receiver's `ViT2` opens with `nn.BatchNorm2d(3)` over
    raw RGB and ShapeWorld gets no other input normalisation, so what this
    function emits lands directly on that layer's statistics -- and eval is never
    silhouetted, so the running stats are gathered on a mixture and then used on
    clean images. The palette is blue-deficient (blue appears in two of six
    colours where red and green appear in three), so its mean is (148.83, 148.83,
    106.33) and no achromatic constant can match all three channels: 0.5 ran
    -14.3% / -14.3% / +19.9% against eval, where this fill runs +0.1% / +0.1% /
    -0.3%. That is a constant fitted to this dataset rather than the
    maximum-entropy answer for a palette the code does not know, which is what
    0.5 was; the trade is deliberate. See DEFAULT.toml.

    `fill * 255` is still required to be an integer per channel, for a weaker
    reason than it had under coverage blending: there, it was what stopped
    anti-aliased edges landing on colour-dependent rounding ties, and there are
    no anti-aliased edges now. Here it means only that the fill is a colour the
    store can represent exactly rather than one that rounds to a neighbour.
    `tests/test_silhouette.py::test_the_fill_is_an_integer_number_of_levels`
    keeps it honest.

    No palette colour is a fixed point. `0.5 * 255 = 128` is exactly ShapeWorld's
    `gray`, so under the old flat half a grey object came back bit-identical and
    one colour in six was silently exempt from the intervention. A chromatic fill
    has no fixed point at all.

    Assumes one object on a black ground. Two objects of different colours would
    threshold against a peak set by the brighter, and the darker can fall under
    it and be erased outright -- which is what this did before e884662 and does
    again. See docs/dubious-claims.md.

    See docs/data.md.
    """
    if imgs.shape[1] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(imgs.shape)}")

    luma = (imgs.float() * _LUMA.to(imgs.device).view(1, 3, 1, 1)).sum(1)
    # Per image, so that a dim object binarises the same way a bright one does.
    peak = luma.amax(dim=(1, 2), keepdim=True)
    # An all-black image has peak 0 and must stay black rather than turn grey.
    on = (luma > peak / 2) & (peak > 0)

    max_value = 255 if not imgs.dtype.is_floating_point else 1.0
    # A tensor before any arithmetic touches it: `(0.5, 0.5, 0.4) * 255` is
    #     tuple repetition, not a scaled colour.
    out = on.unsqueeze(1) * (_as_fill(fill, imgs.device) * max_value)

    # Integer dtypes truncate on cast, and an arbitrary `fill` need not be an
    #     integer number of levels even though the default is.
    if not imgs.dtype.is_floating_point:
        out = out.round()

    # No `expand_as`: the (1, 3, 1, 1) fill above already broadcast the single
    #     mask channel to three, and with a chromatic fill they differ.
    return out.to(imgs.dtype).contiguous()


class ConceptDataset:
    def __init__(
        self,
        data,
        vocab,
        n_examples=None,
        augment=False,
        reference_game=False,
        percent_novel=1.0,
        name=None,
        visfunc=vis_image,
        image_size=None,
        silhouette_p_sender=0.0,
        silhouette_p_receiver=0.0,
        silhouette_fill=DEFAULT_SILHOUETTE_FILL,
        augment_flip=False,
        augment_affine_degrees=0.0,
        **kwargs,
    ):
        self.x = data["x"]
        self.n_feats = self.x[0].shape[1:]
        self.n_examples = n_examples
        self.image_size = image_size

        self.name = name

        self.labels = data["labels"]
        self.lang_raw = data["langs"]
        self.metadata = data["metadata"]
        self.augment = augment
        # Get vocab
        self.vocab = vocab
        self.w2i = vocab["w2i"]
        self.i2w = vocab["i2w"]
        self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)
        self.vis_input = visfunc
        self.reference_game = reference_game
        self.percent_novel = percent_novel
        self.silhouette_p_sender = silhouette_p_sender
        self.silhouette_p_receiver = silhouette_p_receiver
        self.silhouette_fill = silhouette_fill
        self.augment_flip = augment_flip
        self.augment_affine_degrees = augment_affine_degrees
        assert self.n_examples % 2 == 0
        # Assign the rest of the kwargs
        for name, val in kwargs.items():
            if hasattr(self, name):
                raise ValueError(f"Received > 1 argument for {name}")
            setattr(self, name, val)

    def __len__(self):
        return len(self.lang_raw)

    def _apply_silhouette(self, spk_inp, lis_inp):
        """
        Silhouette each agent's whole view, or neither, per game. The two rolls
        are independent. See docs/data.md.
        """
        if self.silhouette_p_sender and np.random.rand() < self.silhouette_p_sender:
            spk_inp = silhouette(spk_inp, self.silhouette_fill)
        if self.silhouette_p_receiver and np.random.rand() < self.silhouette_p_receiver:
            lis_inp = silhouette(lis_inp, self.silhouette_fill)
        return spk_inp, lis_inp

    def _augment_geometry(self, imgs):
        """
        Flip and rotate each of one agent's referents, independently.

        Per *image* and not per game, and called once per agent rather than
        once on the whole row. Both matter. A single draw applied to the whole
        tensor would leave every referent in the game -- and both agents' views
        of the shared stored image -- under the same transform, which varies
        the epoch but not the game. Drawing per image means the listener never
        sees the same pixel array twice, which is the point: the store holds 20
        positives per game and a hundred epochs of the same twenty is what a
        listener memorises. See docs/data.md.

        Safe against this dataset's five shapes -- circle, ellipse, rectangle,
        square, triangle -- which is not a property of affine transforms in
        general and is why `translate`, `scale` and `shear` are pinned off
        rather than left to a config:

        - Flips alias nothing. A flipped triangle is still a triangle; there is
          no inverted-triangle label.
        - Rotation aliases nothing at small angles. The dangerous one is 45
          degrees on a square, and there is no diamond label anyway.
        - *Shear* would alias. A sheared rectangle is a parallelogram and a
          sheared square stops being square.
        - *Anisotropic* scaling would alias two label pairs outright:
          circle-ellipse and square-rectangle. `RandomAffine`'s own `scale` is
          isotropic and so would be safe, but nothing here needs it.

        `fill=0` because ShapeWorld renders on a black background (see
        `silhouette`), so the corners rotation leaves behind are the background
        colour rather than a value that appears nowhere else in the dataset and
        which a model could key on.

        Applied after `_apply_silhouette`, not before: the silhouette thresholds
        stored pixel values, and interpolation would blur exactly the edges that
        threshold reads.
        """
        if not (self.augment_flip or self.augment_affine_degrees):
            return imgs

        out = imgs.clone()
        for j in range(out.shape[0]):
            img = out[j]
            if self.augment_flip:
                # Independent draws, so a quarter of images get both.
                if np.random.rand() < 0.5:
                    img = TF.hflip(img)
                if np.random.rand() < 0.5:
                    img = TF.vflip(img)
            if self.augment_affine_degrees:
                # Every image, not a fraction of them: a rotation of zero is
                #     already in the range, so a probability here would only
                #     concentrate mass on the identity.
                img = TF.affine(
                    img,
                    angle=float(
                        np.random.uniform(
                            -self.augment_affine_degrees,
                            self.augment_affine_degrees,
                        )
                    ),
                    translate=[0, 0],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            out[j] = img
        return out

    def _game_language(self, i, pos_i):
        return self.lang_idx[i]

    def _game_percent_novel(self):
        return self.percent_novel

    @util.return_index
    def __getitem__(self, i):
        img = self.x[i]
        label = self.labels[i]
        md = self.metadata[i]

        assert img.shape[0] % 2 == 0
        midp = img.shape[0] // 2

        # Assert that the positives and negatives look right
        assert np.all(label[:midp])
        assert np.all(~label[midp:])

        if self.reference_game or self.augment:
            # The shuffles below write in place, and with the store in memory
            # `self.x[i]` is a view onto the shared array. See docs/data.md.
            img = np.array(img)

        pos_i = 0

        if self.reference_game:
            # Choose a single random target
            if self.augment:
                pos_i = np.random.randint(midp)
            # Re-assign positive examples
            img[:midp] = img[pos_i]

        # After the target is chosen, before the shuffles: a reference game's
        # language depends on which image was picked.
        lang = self._game_language(i, pos_i)

        if self.augment:
            # Shuffle positives by themselves
            pos_order = np.random.permutation(midp)
            img[:midp] = img[:midp][pos_order]
            # Shuffle negatives by themselves
            neg_order = np.random.permutation(midp)
            img[midp:] = img[midp:][neg_order]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        if self.image_size is not None and self.image_size != img.shape[2]:
            img = F.interpolate(img, (self.image_size, self.image_size))

        spk_inp, spk_label, lis_inp, lis_label = util.split_spk_lis(
            img, label, self.n_examples, percent_novel=self._game_percent_novel()
        )
        spk_inp, lis_inp = self._apply_silhouette(spk_inp, lis_inp)

        # Train only, like the permutation above and the silhouette, and drawn
        #     separately for each agent so that the two views of a shared stored
        #     image diverge.
        if self.augment:
            spk_inp = self._augment_geometry(spk_inp)
            lis_inp = self._augment_geometry(lis_inp)

        return (spk_inp, spk_label, lis_inp, lis_label, lang, md)

    def to_text(self, idxs, join=True):
        def tokenize(lang):
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[language.PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, language.UNK_TOKEN))
            return toks

        return language.rows_to_text(idxs, tokenize, join=join)

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=int) + 2
        lang_idx = np.full(
            (len(langs), max(lang_len)), self.w2i[language.PAD_TOKEN], dtype=int
        )
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[language.SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[language.UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[language.EOS_TOKEN]
        return lang_idx, lang_len
