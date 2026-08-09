import numpy as np
import os
import gzip
import h5py

import torch
import torch.nn.functional as F

import json

from . import generic
from . import language
from . import util


SHAPES = ["circle", "ellipse", "square", "rectangle", "triangle"]
COLORS = ["red", "blue", "green", "yellow", "white", "gray"]
SHAPES_DICT = {v: k for k, v in enumerate(SHAPES)}
COLORS_DICT = {v: k for k, v in enumerate(COLORS)}
N_FEATS = len(SHAPES) + len(COLORS)


# Only the splits the run actually consumes. There is no val split because there
# is no best-epoch selection: training runs to a fixed endpoint and the
# per-epoch metrics.csv trajectory is the deliverable. `_same` splits are
# optional (see the tolerance in `load`).
SPLITS = ["train", "test", "test_same"]


def _concept_to_lf(concept):
    """
    Parse concept. Since fixed recursion we don't have to worry about
    precedence
    """
    op = ""
    if "or" in concept:
        op = "or"
    elif "and" in concept:
        op = "and"

    if op:
        op_index = concept.index(op)
        left = concept[:op_index]
        right = concept[op_index + 1 :]
        return (op, _concept_to_lf(left), _concept_to_lf(right))

    if "not" in concept:
        assert len(concept) == 2, f"unable to parse {concept}"
        return ("not", (concept[1],))

    assert len(concept) == 1, f"unable to parse {concept}"
    return (concept[0],)


def concept_to_lf(concept, split=True):
    if split:
        concept = concept.split(" ")
    return _concept_to_lf(concept)


def extract_shapes(worlds):
    """
    Turn worlds into one-hot arrays.
    """
    shapes = []

    for world in worlds:
        imgs = world["imgs"]
        this_world_shapes = []

        if len(imgs[0]) > 1:
            raise RuntimeError("More than one shape in this world")
        for img in imgs:
            shape = img[0]
            this_world_shapes.append([shape["color"], "and", shape["shape"]])

        shapes.append(this_world_shapes)

    return shapes


def load(config, fast=False):
    datas = {}
    # if config['sender']['arguments']['image_encoder'] == "PretrainedResNet18":
    #     # Need larger images
    #     image_size = 224
    # else:
    image_size = 64

    for split in SPLITS:
        sfile = os.path.join(config['data']['dataset'], f"{split}.npz")
        sfile_hdf5 = sfile.replace(".npz", ".hdf5")
        is_present = os.path.exists(sfile) or os.path.exists(sfile_hdf5)
        if not is_present:
            if not split.endswith("_same"):
                # Then this split should be here
                raise RuntimeError(f"Can't find {sfile} or {sfile_hdf5}")
            else:
                continue
        datas[split] = load_split(
            config['data']['dataset'],
            split,
            fast=fast,
            into_memory=config['data']['load_shapeworld_into_memory'],
            # `extract_shapes` returns per-*image* descriptors, which disagree
            # with a subsampled (40-image) store. Only reference games consume
            # them, and those use the separate `shapeworld_ref` dataset.
            need_shapes=config['reference_game'],
        )

    langs = np.concatenate([datas[s]["langs"] for s in datas])
    vocab = language.init_vocab(langs)

    # Compute vocab first
    _, md_vocab = get_metadata(langs)

    dataset_kwargs = {
        "n_examples": config['data']['n_examples'],
        "visfunc": generic.vis_image,
        "name": "shapeworld",
        "image_size": image_size,
    }

    datasets = {}
    for split in datas:
        datas[split]["metadata"] = get_metadata(datas[split]["langs"], md_vocab)[0]
        datasets[split] = ShapeWorldDataset(
            datas[split],
            vocab,
            augment=split == "train",
            percent_novel=config['data']['percent_novel'],
            reference_game=config['reference_game'],
            # Training-time only. Eval is never silhouetted, so the reported
            # numbers stay comparable to the paper's and to the `probe_shape.py`
            # sweep, which measures the sender on un-augmented images.
            silhouette_p_sender=(
                config['data']['silhouette_p_sender'] if split == "train" else 0.0
            ),
            silhouette_p_receiver=(
                config['data']['silhouette_p_receiver'] if split == "train" else 0.0
            ),
            shapes=datas[split]["shapes"],
            metadata_vocab=md_vocab,
            **dataset_kwargs,
        )

    # No cross-game-type eval datasets. The run trains and evaluates a single
    # game framing (concept, i.e. `percent_novel = 1.0`), under which the
    # speaker and listener see fully disjoint targets *and* distractors. That
    # disjointness is an intrinsic control against context-dependent degenerate
    # codes, which is what the cross-eval passes were guarding against, so
    # building 12 extra eval datasets bought nothing but I/O.
    return datasets


def load_split(dataset, split, fast=False, into_memory=False, need_shapes=False):
    data_file = os.path.join(dataset, f"{split}.npz")
    if os.path.exists(data_file):
        data = np.load(data_file)
    else:
        # Try hdf5
        data = h5py.File(data_file.replace(".npz", ".hdf5"), "r")
    # `extract_shapes` is per *image*, so it only agrees with the store when the
    # images have not been subsampled; it is consumed solely by
    # `get_reference_game`/`shapes_to_idx`, which concept games never call. It is
    # now the only consumer of the world files, so concept games skip parsing
    # them altogether.
    if fast or not need_shapes:
        shapes = None
    else:
        world_file = os.path.join(dataset, f"{split}_worlds.json")
        if os.path.exists(world_file):
            with open(world_file, "r") as f:
                worlds = json.load(f)
        else:
            with gzip.open(world_file + ".gz", "r") as f:
                worlds = json.load(f)
        shapes = extract_shapes(worlds)

    imgs = data["imgs"]
    labels = data["labels"]
    if into_memory:
        imgs = imgs[:]
        labels = labels[:]

    # hdf5 hands back bytes; npz hands back numpy unicode scalars. Dispatch per
    # element rather than on the array dtype, which reports neither cleanly.
    langs_decoded = [
        lang.decode("utf-8") if isinstance(lang, bytes) else str(lang)
        for lang in data["langs"]
    ]

    # Force 1D object array
    langs = np.empty(len(langs_decoded), dtype=object)
    langs[:] = [t.lower().split() for t in langs_decoded]

    return {
        "x": imgs,
        "labels": labels,
        "langs": langs,
        "shapes": shapes,
    }


def feature_type(feat):
    if feat in COLORS:
        return "color"
    elif feat in SHAPES:
        return "shape"
    else:
        raise ValueError(f"Unknown feature type {feat}")


def get_metadata(langs, md_vocab=None):
    md = []
    if md_vocab is None:
        md_vocab = {
            "w2i": {},
            "i2w": {},
        }
    for lang in langs:
        lc = concept_to_lf(lang, split=False)
        if len(lc) == 1:
            # Single feature. Ignore NOTs (treat them the same)
            this_md = feature_type(lc[0])
            pass
        elif len(lc) == 2:
            # NOT
            this_md = feature_type(lc[1][0])
            pass
        elif len(lc) == 3:
            op = lc[0]
            if len(lc[1]) == 1:
                l_md = feature_type(lc[1][0])
            else:
                l_md = feature_type(lc[1][1][0])
            if len(lc[2]) == 1:
                r_md = feature_type(lc[2][0])
            else:
                r_md = feature_type(lc[2][1][0])
            this_md = f"{op}_{l_md}_{r_md}"
        else:
            raise ValueError(f"Unknown feature type {this_md}")
        if this_md not in md_vocab["w2i"]:
            md_i = len(md_vocab["w2i"])
            md_vocab["w2i"][this_md] = md_i
            md_vocab["i2w"][md_i] = this_md
        else:
            md_i = md_vocab["w2i"][this_md]

        md.append(md_i)
    return md, md_vocab


class ShapeWorldDataset(generic.ConceptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tokenize metadata language
        if self.shapes is not None:
            self.shape_lang_idx, self.shape_lang_len = self.shapes_to_idx()

    @util.return_index
    def get_reference_game(self, i):
        # Copied because the re-assignment/shuffles below write in place, and
        # with the store in memory `self.x[i]` is a view onto the shared array
        # (see the matching note in `generic.ConceptDataset.__getitem__`).
        img = np.array(self.x[i])
        label = self.labels[i]
        md = self.metadata[i]

        midp = img.shape[0] // 2

        # Choose a single random target
        if self.augment:
            pos_i = np.random.randint(midp)
        else:
            pos_i = 0

        # lang to be the shape of the positive target
        lang = self.shape_lang_idx[i, pos_i]
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
            img, label, self.n_examples, percent_novel=0.0
        )
        # `percent_novel = 0.0` hands back the *same* tensor for both agents;
        # `silhouette` returns a new one, so an independent roll per agent is
        # still safe here.
        spk_inp, lis_inp = self._apply_silhouette(spk_inp, lis_inp)
        return (spk_inp, spk_label, lis_inp, lis_label, lang, md)

    def __getitem__(self, i):
        if self.reference_game:
            return self.get_reference_game(i)
        else:
            return super().__getitem__(i)

    def shapes_to_idx(self):
        n = len(self.shapes)
        n_img = len(self.shapes[0])
        shape_lang_len = np.full((n, n_img), 5, dtype=int)
        shape_lang_idx = np.zeros((n, n_img, 5), dtype=int)
        for i in range(n):
            for j in range(n_img):
                shape_lang_idx[i, j, 0] = self.w2i[language.SOS_TOKEN]
                for tok_i, tok in enumerate(self.shapes[i][j], start=1):
                    shape_lang_idx[i, j, tok_i] = self.w2i.get(
                        tok, self.w2i[language.UNK_TOKEN]
                    )
                shape_lang_idx[i, j, -1] = self.w2i[language.EOS_TOKEN]
        return shape_lang_idx, shape_lang_len
