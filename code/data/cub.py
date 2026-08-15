import hashlib
import numpy as np
import os
from pathlib import Path
import torch
from PIL import Image
import pandas as pd


from . import util
from . import image_util as iu


IMAGE_SIZE = 224
LOAD_INTO_MEMORY = True

# CUB classes are 1-indexed. 1-150 are the training species; 151-200 are held out
# wholesale and are the novel-concept `test` split -- the same 50 species as
# jayelm's, so the generalisation number keeps its old footing.
#
# Classes 101-150 were a val split. There is no val split any more (no best-epoch
# selection), so rather than sit unused they are folded into training. That is
# what pays for the image-level holdout below: 150 species at 80% of their images
# is ~60% of the corpus for training, against the ~50% that 100 whole species
# gave, with half again as much species diversity.
TRAIN_CLASSES = range(1, 151)
TEST_CLASSES = range(151, 201)

# jayelm's was `range(4)`, which asked for a class 0 that does not exist and so
# gave debug runs three species rather than four.
TRAIN_CLASSES_DEBUG = range(1, 5)
TEST_CLASSES_DEBUG = range(8, 12)

# The `test_same` holdout. ShapeWorld gets its seen-concept split for free --
# `test_same.npz` holds freshly generated worlds, so they are unseen *images* of
# seen *concepts* -- but CUB has a finite pool of photographs per species, so the
# same property has to be bought by holding images out of training. Without it
# `test_same` would measure recall of images the sender had already trained on
# rather than generalisation to new instances of a seen concept, which is what
# the paper's Acc (Seen) column reports.
#
# These are module constants rather than config keys on purpose. Changing either
# moves the boundary between `train` and `test_same`, which invalidates every run
# recorded under the old value; that is a version of the dataset, not a knob to
# turn per run. Bump the salt's suffix if the partition ever has to change, so
# that old and new runs are distinguishable rather than silently pooled.
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
    A stable pseudo-random sort key for one image name.

    `hashlib` rather than an RNG, and blake2b rather than the builtin `hash()`,
    because both of the alternatives move. `hash()` is salted per process, and
    `numpy.random.Generator`'s stream is explicitly not guaranteed across numpy
    releases (NEP 19 freezes only the legacy `RandomState`). A hash function is
    fixed by its algorithm, so this partition reproduces on any machine, in any
    environment, under any future version.
    """
    return hashlib.blake2b(
        f"{HOLDOUT_SALT}:{name}".encode("utf-8"), digest_size=16
    ).digest()


def holdout_image_names(img_names, n_examples, fraction=HOLDOUT_FRACTION):
    """
    Choose the `test_same` images, per species.

    `img_names` maps class id -> every image name of that species; the return
    maps class id -> the frozenset of names held out of training.

    This function touches no RNG at all, global or local, and that is a
    requirement rather than a nicety. `train.py` seeds numpy from `--seed` and
    `loader.worker_init` re-seeds it per worker, so an `np.random.choice` here
    would hand every seed -- and every resume, and every dataloader worker -- a
    different partition, and an image held out under seed 0 would be trained on
    under seed 1. Since the sort key depends on the image *name* alone, the
    result is also independent of how many species are passed, of the order they
    are passed in, and of npz key order (which is whatever `os.listdir` happened
    to hand `save_cub_np.py`).

    Size is `max(n_examples, round(fraction * n))`. CUB species carry 41-60
    images, so 20% is 8-12 and the floor binds only at the bottom of that range.
    The floor is required: `sample_game` draws `n_examples` *distinct* positives
    from a single species per game (`replace=False`), so a pool of 8 would raise
    on every game played on that species. `fraction * n` cannot land on a .5 for
    integer `n` at 0.2, so the rounding rule is not load-bearing.
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
    *distractors* held out too. `sample_negatives` can only reach what is in
    `self.imgs`, so a dataset built from the held-out pool cannot show the
    listener an image the sender trained on, on either side of the game.
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
    How many games one epoch of `split` draws.

    Train: `CUBDataset.__getitem__` ignores its index and samples a fresh game,
    so this is the size of an epoch rather than a set of stored games.
    Consecutive epochs are independent draws from a combinatorially large space,
    and nothing here is exhausted by raising it. See `games_per_epoch` in
    `[birds.data]` for why the default is no longer jayelm's 1,000.

    Eval: sized *per species* rather than flat. The two eval splits hold 50 and
    150 species, so one shared game count would give them very different
    per-concept coverage -- and topsim measures one prototype per concept, built
    from the modal message over that concept's instances, so coverage per species
    is the quantity that has to be held equal for the two splits to be read
    against each other. jayelm's flat 200 gave `test` four games a species and
    would have given `test_same` 1.3.
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
        # if LOAD_INTO_MEMORY:  # Load npz into memory
        #     cl_imgs = dict(cl_imgs)
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

    # Only the splits `train.py` consumes. `train` and `test_same` are the two
    # sides of one image-level partition of the *same* 150 species, so
    # `test_same` is the paper's Acc (Seen): held-out photographs of concepts the
    # sender was trained on. `test` species are unseen wholesale, so no
    # image-level holdout applies to them.
    #
    # There is no val split (no best-epoch selection) and no cross-game-type eval
    # datasets: the run trains and evaluates a single game framing, so the
    # `<split>_<game_type>` grid that used to be built here was never read. See
    # the matching note in `shapeworld.load`.
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
        """
        Get an item. Note the i doesn't matter, we just randomly sample.
        (Should this be the case for val? maybe not?)
        """
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
        texts = []
        for lang in idxs:
            toks = []
            toks.append("<s>")
            for i in lang[1:-1]:
                toks.append(str(i.item()))
            toks.append("</s>")
            if join:
                texts.append(" ".join(toks))
            else:
                texts.append(toks)
        return texts
