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


def silhouette(imgs):
    """
    Render each image as a white-on-black silhouette of its object.

    ShapeWorld's six colours (red, blue, green, yellow, white, gray) sit at six
    distinct luma values -- roughly 29, 76, 128, 150, 226, 255 -- so a plain
    grayscale conversion does not remove colour, it re-encodes it as a single
    scalar that one conv filter can threshold. Thresholding does remove it: with
    a single object on a black ground every colour renders identically, so the
    mutual information between colour and pixels is zero by construction rather
    than by hoping two distributions overlap.

    The threshold is half of each image's own peak luma. A fixed threshold would
    not do, because blue peaks at 0.114 while white peaks at 1.0; taking it
    relative to the peak binarises both, and puts the cut below the anti-aliased
    edge pixels either way.

    `imgs` is (n, C, H, W) with a black background. dtype and range are
    preserved, so uint8 images come back in {0, 255} and float images in
    {0.0, 1.0}. Returns a new tensor; the input is left untouched.
    """
    if imgs.shape[1] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(imgs.shape)}")

    luma = (imgs.float() * _LUMA.to(imgs.device).view(1, 3, 1, 1)).sum(1)
    # Per image, so that a dim object binarises the same way a bright one does.
    peak = luma.amax(dim=(1, 2), keepdim=True)
    # An all-black image has peak 0 and must stay black rather than turn white.
    on = (luma > peak / 2) & (peak > 0)

    on_value = 255 if not imgs.dtype.is_floating_point else 1.0
    return (on.unsqueeze(1) * on_value).to(imgs.dtype).expand_as(imgs).contiguous()


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
        Silhouette each agent's whole view, or neither, per game.

        The roll is per *game*, not per image: with 10 targets in a set, rolling
        per image would leave ~(1-p) x 10 of them coloured and the colour cue
        recoverable from the set, which is not the intervention. Silhouetting
        the whole view is what makes shape the only available cue.

        The two rolls are independent, so the pair of rates selects the regime:
        (0, p) silhouettes only the receiver, (p, 0) only the sender, (p, p)
        either or both. Each agent's view is a full side of the game, so one
        roll already covers both polarities and, in the concept game, the
        disjoint half that agent sees.
        """
        if self.silhouette_p_sender and np.random.rand() < self.silhouette_p_sender:
            spk_inp = silhouette(spk_inp)
        if self.silhouette_p_receiver and np.random.rand() < self.silhouette_p_receiver:
            lis_inp = silhouette(lis_inp)
        return spk_inp, lis_inp

    @util.return_index
    def __getitem__(self, i):
        img = self.x[i]
        label = self.labels[i]
        lang = self.lang_idx[i]
        md = self.metadata[i]

        assert img.shape[0] % 2 == 0
        midp = img.shape[0] // 2

        # Assert that the positives and negatives look right
        assert np.all(label[:midp])
        assert np.all(~label[midp:])

        if self.reference_game or self.augment:
            # The shuffles below write in place. When the store is in memory
            # (`load_shapeworld_into_memory`), `self.x[i]` is a *view* onto the
            # shared array, so writing through it would permanently permute the
            # dataset -- and, worse, break copy-on-write for forked dataloader
            # workers, which would each end up copying every row they touch.
            # Copy the row instead; it is ~0.5 MB and costs nothing next to the
            # forward pass.
            img = np.array(img)

        if self.reference_game:
            # Choose a single random target
            if self.augment:
                pos_i = np.random.randint(midp)
            else:
                pos_i = 0
            # Re-assign positive examples
            img[:midp] = img[pos_i]

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
            img, label, self.n_examples, percent_novel=self.percent_novel
        )
        spk_inp, lis_inp = self._apply_silhouette(spk_inp, lis_inp)
        return (spk_inp, spk_label, lis_inp, lis_label, lang, md)

    def to_text(self, idxs, join=True):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[language.PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, language.UNK_TOKEN))
            if join:
                texts.append(" ".join(toks))
            else:
                texts.append(toks)
        return texts

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
