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


# Only the splits the run consumes; `_same` splits are optional. See docs/data.md.
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
            # with a subsampled store. See docs/data.md.
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
            # Training-time only; eval is never silhouetted. See docs/data.md.
            silhouette_p_sender=(
                config['data']['silhouette_p_sender'] if split == "train" else 0.0
            ),
            silhouette_p_receiver=(
                config['data']['silhouette_p_receiver'] if split == "train" else 0.0
            ),
            # Not split-gated: the fill only has an effect where a rate is
            # non-zero, and eval's rates are already 0.0 above.
            silhouette_fill=config['data']['silhouette_fill'],
            shapes=datas[split]["shapes"],
            metadata_vocab=md_vocab,
            **dataset_kwargs,
        )

    return datasets


def load_split(dataset, split, fast=False, into_memory=False, need_shapes=False):
    data_file = os.path.join(dataset, f"{split}.npz")
    if os.path.exists(data_file):
        data = np.load(data_file)
    else:
        # Try hdf5
        data = h5py.File(data_file.replace(".npz", ".hdf5"), "r")
    # Only reference games consume the world files. See docs/data.md.
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

    def _game_language(self, i, pos_i):
        """The target image's shape, not the concept: the positives are copies."""
        if self.reference_game:
            return self.shape_lang_idx[i, pos_i]
        return super()._game_language(i, pos_i)

    def _game_percent_novel(self):
        """
        0.0 hands back the *same* tensor to both agents; `silhouette` returns a
        new one, so the independent rolls downstream are still safe.
        """
        if self.reference_game:
            return 0.0
        return super()._game_percent_novel()

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
