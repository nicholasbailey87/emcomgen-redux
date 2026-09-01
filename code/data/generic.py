import numpy as np
import torch
import torch.nn.functional as F
import os

from PIL import Image
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


# ShapeWorld's own mean object colour, (149, 149, 106) of 255. Chromatic, and
#     an integer number of levels in every channel -- both load-bearing, and
#     both explained under `silhouette` and in docs/data.md. This was a flat 0.5
#     until 2026-09-01, which collided with the palette's `gray` and left a
#     measurable colour leak in the anti-aliased edges.
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

    `imgs` is (n, C, H, W) with a black background. Every pixel is scaled by the
    object's coverage there, so a fully covered pixel comes back at `fill` of
    maximum intensity, the background stays at zero, and an anti-aliased edge
    keeps the coverage it had. dtype and range are preserved -- `fill` is read
    against 255 for integer images and against 1.0 for floating-point ones.
    Returns a new tensor; the input is left untouched.

    `fill` is a length-3 per-channel fraction, or a scalar broadcast to three.

    Coverage rather than a threshold. For a flat object on black, a fully
    covered pixel has `luma == peak` and a pixel at coverage `k` has
    `luma == k * peak`, so `luma / peak` recovers the coverage exactly and does
    so independently of the object's colour -- the same colour-invariance the
    old `luma > peak / 2` threshold had, without discarding the edges on the way
    through. The threshold promoted every partly-lit edge pixel to fully lit,
    which made the repainted region larger than the object it replaced.

    Chromatic, and an integer number of levels. The default is the palette's own
    mean object colour, (149, 149, 106). Two things follow, and both are the
    reason it is not a flat half.

    First, `fill * 255` is an integer per channel, and that is what makes the
    edges colour-invariant. A stored edge pixel is `round(coverage * fill *
    255)`; at `fill = 0.5` that is `coverage * 127.5`, and a bright colour's
    coverage is exactly `n/255`, so the product is exactly `n/2` and every odd
    `n` lands on a rounding tie. Ties break on the last bits of a float32 whose
    value depends on the colour's luma weights, so 43 of the 256 stored levels
    made red, blue, green, yellow and white disagree -- a colour signal in the
    anti-aliasing. With `fill * 255` an integer `F`, `n * F / 255` cannot be a
    half-integer (255 is odd, `2nF` is even), so no *cross-colour* tie exists.
    Grey still has ties of its own -- its coverage is `m/128`, so `m = 64` gives
    74.5 -- but grey is alone on that ramp and a tie there makes nothing
    disagree with anything.

    Second, no palette colour is a fixed point any more. `0.5 * 255 = 128` is
    exactly ShapeWorld's `gray`, so a grey object came back bit-identical and
    one colour in six was silently exempt from the intervention.

    The receiver's `ViT2` opens with `nn.BatchNorm2d(3)` over raw RGB and
    ShapeWorld gets no other input normalisation, so what this function emits
    lands directly on that layer's statistics -- and eval is never silhouetted,
    so the running stats are gathered on a mixture and then used on clean
    images. The palette is blue-deficient (blue appears in two of six colours
    where red and green appear in three), so its mean is (148.83, 148.83,
    106.33) and no achromatic constant can match all three channels: 0.5 ran
    -14.3% / -14.3% / +19.9% against eval, where this fill runs +0.1% / +0.1% /
    -0.3%. That is a constant fitted to this dataset rather than the
    maximum-entropy answer for a palette the code does not know, which is what
    0.5 was; the trade is deliberate. See DEFAULT.toml.

    A leak remains, and it is not fixable here. An edge pixel is stored as
    `round(k * C)` per channel, so the number of representable coverage levels
    is `255 * V + 1` for `V = max(R, G, B) / 255`. Every palette colour has
    V = 1 except grey, at V = 0.502, so grey resolves coverage in 129 steps
    where the rest use 256 and its output skips specific intensity values --
    a structural gap in the value histogram that a classifier finds at ~0.97
    recall against a chance of 0.167. See docs/data.md and docs/dubious-claims.md.

    Assumes one object on a black ground, as the old threshold did. Two objects
    of different colours would come back at different intensities, which leaks
    their relative luma -- though it leaks less than the threshold did, which
    erased the darker of the two outright.

    See docs/data.md.
    """
    if imgs.shape[1] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(imgs.shape)}")

    luma = (imgs.float() * _LUMA.to(imgs.device).view(1, 3, 1, 1)).sum(1)
    # Per image, so that a dim object repaints the same way a bright one does.
    peak = luma.amax(dim=(1, 2), keepdim=True)
    # An all-black image has peak 0 and must stay black rather than turn grey,
    #     and dividing by it would be a NaN rather than a zero.
    lit = peak > 0
    safe_peak = torch.where(lit, peak, torch.ones_like(peak))
    coverage = torch.where(lit, luma / safe_peak, torch.zeros_like(luma))

    max_value = 255 if not imgs.dtype.is_floating_point else 1.0
    # A tensor before any arithmetic touches it: `(0.5, 0.5, 0.4) * 255` is
    #     tuple repetition, not a scaled colour.
    out = coverage.unsqueeze(1) * (_as_fill(fill, imgs.device) * max_value)

    # Integer dtypes truncate on cast, so `0.999 * 128` would land at 127.
    if not imgs.dtype.is_floating_point:
        out = out.round()

    # No `expand_as`: the (1, 3, 1, 1) fill above already broadcast the single
    #     coverage channel to three, and with a chromatic fill they differ.
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
