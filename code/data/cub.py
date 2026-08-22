import hashlib
import numpy as np
import os
from pathlib import Path
import torch
from PIL import Image
import pandas as pd


from . import language
from . import util
from . import image_util as iu


IMAGE_SIZE = 224
LOAD_INTO_MEMORY = True

# CUB classes are 1-indexed. See docs/data.md.
TRAIN_CLASSES = range(1, 151)
TEST_CLASSES = range(151, 201)

TRAIN_CLASSES_DEBUG = range(1, 5)
TEST_CLASSES_DEBUG = range(8, 12)

# The `test_same` holdout. Module constants rather than config keys on purpose:
# changing either is a new version of the dataset. Bump the salt's suffix if the
# partition ever has to change. See docs/data.md.
HOLDOUT_FRACTION = 0.2
HOLDOUT_SALT = "cub-test_same-v1"


def load_class_metadata(cub_path):
    md_path = str(
        Path(cub_path) / "attributes" / "class_attribute_labels_continuous.txt"
    )
    md = pd.read_csv(md_path, sep=" ", header=None).to_numpy(dtype=np.float32)
    md = (md > 50.0).astype(np.uint8)
    # Now map to unique class names which are +1 indexed
    md_dict = {}
    for i in range(md.shape[0]):
        md_dict[i + 1] = md[i]
    # Now map from image ids to unique class names
    im_path = str(Path(cub_path) / "image_class_labels.txt")
    im2cl = pd.read_csv(im_path, sep=" ", header=None, names=["image_id", "class_id"])
    im2cl = dict(zip(im2cl["image_id"], im2cl["class_id"]))

    im_dict = {}
    for im_id, cl_id in im2cl.items():
        im_dict[im_id] = md_dict[cl_id]
    return im_dict


def load_img_metadata(cub_path):
    md_path = str(Path(cub_path) / "attributes" / "image_attribute_labels.txt")
    md = pd.read_csv(
        md_path,
        sep=" ",
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        usecols=["is_present"],
        dtype={"is_present": np.uint8},
        header=None,
    )
    md_arr = md["is_present"].to_numpy()
    # Take slices by # of attributes, which is by 312
    md_dict = {}
    i = 1  # img id
    for start in range(0, len(md_arr), 312):
        md_dict[i] = md_arr[start : start + 312]
        i += 1

    # Now this is a mapping from each image id to the array
    return md_dict


def load_cub_metadata(config):
    cub_dir = str(Path(config['data']['dataset']) / "CUB_200_2011")

    # Load metadata per image
    img_md = load_img_metadata(cub_dir)
    class_md = load_class_metadata(cub_dir)

    # Load mapping from image names (npz keys) to metadata
    id2name = pd.read_csv(
        str(Path(cub_dir) / "images.txt"),
        sep=" ",
        names=["image_id", "name"],
        header=None,
    )
    ids = id2name["image_id"]
    names = [str(Path(i)) for i in id2name["name"]]
    id2name = dict(zip(ids, names))

    def rename_md(md):
        # Rename to image names
        md = {id2name[i]: m for i, m in md.items()}
        # Add path to cub dir so we can look up md in CUBDataset
        md = {
            str(Path("cub") / "CUB_200_2011" / "images" / k): v for k, v in md.items()
        }
        return md

    img_md = rename_md(img_md)
    class_md = rename_md(class_md)

    return img_md, class_md


def _holdout_rank(name):
    """
    A stable pseudo-random sort key for one image name. A hash rather than an
    RNG so the partition reproduces anywhere; see docs/data.md.
    """
    return hashlib.blake2b(
        f"{HOLDOUT_SALT}:{name}".encode("utf-8"), digest_size=16
    ).digest()


def holdout_image_names(img_names, n_examples, fraction=HOLDOUT_FRACTION):
    """
    Choose the `test_same` images, per species.

    `img_names` maps class id -> every image name of that species; the return
    maps class id -> the frozenset of names held out of training.

    Touches no RNG at all, global or local, and that is a requirement rather
    than a nicety. See docs/data.md.
    """
    holdout = {}
    for cl, names in img_names.items():
        n = len(names)
        k = max(n_examples, int(round(fraction * n)))
        if n - k < n_examples:
            raise ValueError(
                f"class {cl} has {n} images; holding out {k} leaves {n - k} for "
                f"training, below n_examples={n_examples}. Every game needs "
                f"{n_examples} distinct positives from a single species."
            )
        holdout[cl] = frozenset(sorted(names, key=_holdout_rank)[:k])
    return holdout


def split_images(imgs, classes, holdout=None, keep_holdout=False):
    """
    Subset `imgs` (class id -> {image name: array}) to `classes` and, where a
    `holdout` map is given, to one side of that species' holdout.

    Filtering here rather than at sampling time is what makes the `test_same`
    *distractors* held out too. See docs/data.md.
    """
    subset = {}
    for cl, cl_imgs in imgs.items():
        if cl not in classes:
            continue
        if holdout is None:
            subset[cl] = cl_imgs
        else:
            held = holdout[cl]
            subset[cl] = {
                name: img
                for name, img in cl_imgs.items()
                if (name in held) == keep_holdout
            }
    return subset


def n_games(split, n_species, config):
    """
    How many games one epoch of `split` draws. Train counts games, eval counts
    games *per species*. See docs/data.md.
    """
    if split == "train":
        length = config['data'].get('games_per_epoch', 1000)
    else:
        length = config['data'].get('eval_games_per_species', 16) * n_species

    if config['debug']:
        length = max(1, length // 10)

    return length


def load(config):
    img_dir = str(Path(config['data']['dataset']) / "CUB_200_2011" / "images")

    classes = os.listdir(img_dir)
    imgs = {}
    print("Loading CUB...")
    for cl in classes:
        cl_n = int(cl.split(".")[0])
        if config['debug']:
            if not any(
                cl_n in r
                for r in [TRAIN_CLASSES_DEBUG, TEST_CLASSES_DEBUG]
            ):
                continue
        npz_dir = str(Path(img_dir) / cl / "img.npz")
        if not os.path.exists(npz_dir):
            raise RuntimeError(
                f"Couldn't find {npz_dir}, run save_cub_np.py in data/ first?"
            )
        cl_imgs = np.load(npz_dir)
        cl_imgs = dict(cl_imgs)
        cl_imgs = {str(Path(a)): b for a, b in cl_imgs.items()}
        imgs[cl_n] = cl_imgs
    print("...done")

    # Load metadata
    img_md, class_md = load_cub_metadata(config)
    if config['reference_game']:
        md = img_md
    else:
        md = class_md

    tloader = iu.TransformLoader(IMAGE_SIZE)
    train_transform = tloader.get_composed_transform(
        aug=True,
        normalize=True,
        to_pil=True,
    )
    test_transform = tloader.get_composed_transform(
        aug=False,
        normalize=True,
        to_pil=True,
    )

    def to_dset(subset, split):
        return CUBDataset(
            subset,
            md,
            transform=train_transform if split == "train" else test_transform,
            n_examples=config['data']['n_examples'],
            # `len(subset)` rather than the class range, so the eval size follows
            #     the species actually present on disk.
            length=n_games(split, len(subset), config),
            reference_game=config['reference_game'],
            percent_novel=config['data']['percent_novel'],
        )

    if config['debug']:
        train_classes, test_classes = TRAIN_CLASSES_DEBUG, TEST_CLASSES_DEBUG
    else:
        train_classes, test_classes = TRAIN_CLASSES, TEST_CLASSES

    holdout = holdout_image_names(
        {cl: list(cl_imgs) for cl, cl_imgs in imgs.items() if cl in train_classes},
        n_examples=config['data']['n_examples'],
    )

    # Only the splits `train.py` consumes: `train` and `test_same` are the two
    # sides of one image-level partition of the same 150 species. See
    # docs/data.md.
    return {
        "train": to_dset(
            split_images(imgs, train_classes, holdout, keep_holdout=False), "train"
        ),
        "test": to_dset(split_images(imgs, test_classes), "test"),
        "test_same": to_dset(
            split_images(imgs, train_classes, holdout, keep_holdout=True), "test_same"
        ),
    }


class CUBDataset:
    MAX_N_EXAMPLES = 20
    name = "cub"

    def __init__(
        self,
        imgs,
        metadata,
        n_examples=None,
        transform=None,
        length=1000,
        reference_game=False,
        percent_novel=1.0,
    ):
        self.imgs = imgs
        self.metadata = metadata
        self.classes = np.array(list(self.imgs.keys()))
        self.img_names = {c: list(i.keys()) for c, i in self.imgs.items()}
        self.length = length
        self.transform = transform
        self.reference_game = reference_game
        self.n_feats = (3, IMAGE_SIZE, IMAGE_SIZE)
        if n_examples is None:
            self.n_examples = self.MAX_N_EXAMPLES
        else:
            self.n_examples = n_examples
        self.percent_novel = percent_novel
        
        # Make sure the metadata matches up
        for c, ls in self.img_names.items():
            for l in ls:
                assert l in self.metadata, l

    @util.return_index
    def __getitem__(self, i):
        """Note `i` is ignored: every call samples a fresh game."""
        return self.sample_game()

    def sample_negatives(self, n, pos_cl):
        neg_imgs = []
        for _ in range(n):
            neg_cl = pos_cl
            while neg_cl == pos_cl:
                neg_cl = np.random.choice(self.classes)
            neg_img_name = np.random.choice(self.img_names[neg_cl])
            # Choose a cl
            neg_img = self.imgs[neg_cl][neg_img_name]
            neg_imgs.append(neg_img)
        return neg_imgs

    def sample_game(self):
        # Randomly choose a class
        cl = np.random.choice(self.classes)
        if self.reference_game:
            # Select a single positive target
            pos_name = np.random.choice(self.img_names[cl])
            pos_imgs = [self.imgs[cl][pos_name] for _ in range(self.n_examples)]
            md = self.metadata[pos_name]
            percent_novel = 0.0
        else:
            pos_names = np.random.choice(
                self.img_names[cl], size=self.n_examples, replace=False
            )
            pos_imgs = [self.imgs[cl][name] for name in pos_names]
            md = self.metadata[pos_names[0]]
            percent_novel = self.percent_novel

        neg_imgs = self.sample_negatives(self.n_examples, cl)

        if self.transform is not None:
            pos_imgs = [self.transform(img) for img in pos_imgs]
            neg_imgs = [self.transform(img) for img in neg_imgs]
        else:
            # Convert to tensor
            raise NotImplementedError

        imgs, y = util.stack_pos_neg(pos_imgs, neg_imgs)

        # 0th metadata is a game indicator, which we don't use
        md = torch.from_numpy(md)
        diff = torch.zeros((1,), dtype=md.dtype)
        md = torch.cat([diff, md], 0)

        # "padding"
        txt = np.full(3, cl, dtype=np.int64)

        splits = util.split_spk_lis(
            imgs, y, self.n_examples, percent_novel=percent_novel
        )

        return splits + (txt, md)

    def __len__(self):
        return self.length

    def vis_input(self, inp, overwrite=True, **kwargs):
        img_fname = f"{kwargs['name']}_{kwargs['epoch']}_{kwargs['split']}_{kwargs['game_i']}_{kwargs['i']}.jpg"
        img_f = str(Path(kwargs["exp_dir"]) / "images" / img_fname)
        img_html = f"""<img src="{str(Path('images') / img_fname)}">"""
        if os.path.exists(img_f) and not overwrite:
            return img_html
            return
        inp = iu.unnormalize_t_(inp).permute((1, 2, 0)).numpy()
        inp = np.round(inp * 255).astype(np.uint8)
        Image.fromarray(inp).save(img_f)
        return img_html

    def to_text(self, idxs, join=True):
        """A CUB "sentence" is the class id repeated: nothing to look up."""
        def tokenize(lang):
            return (
                [language.SOS_TOKEN]
                + [str(i.item()) for i in lang[1:-1]]
                + [language.EOS_TOKEN]
            )

        return language.rows_to_text(idxs, tokenize, join=join)
