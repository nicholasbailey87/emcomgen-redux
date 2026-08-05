import torch


def stack_pos_neg(pos_imgs, neg_imgs):
    """
    Given positive images and negative images, stack them, then create a y tensor of labels.
    """
    pos_imgs = torch.stack(pos_imgs)
    neg_imgs = torch.stack(neg_imgs)

    imgs = torch.cat([pos_imgs, neg_imgs], 0)
    y = torch.zeros(imgs.shape[0], dtype=torch.uint8)
    y[: pos_imgs.shape[0]] = 1
    return imgs, y


def split_spk_lis(inp, y, n_examples, percent_novel=1.0):
    midp = inp.shape[0] // 2
    n_pos_ex = n_examples // 2

    spk_inp = torch.zeros((n_examples, *inp.shape[1:]), dtype=inp.dtype)
    spk_inp[:n_pos_ex] = inp[:n_pos_ex]
    spk_inp[n_pos_ex:] = inp[midp : midp + n_pos_ex]

    spk_label = torch.zeros(n_examples, dtype=torch.uint8)
    spk_label[:n_pos_ex] = 1

    lis_inp = torch.zeros((n_examples, *inp.shape[1:]), dtype=inp.dtype)
    lis_inp[:n_pos_ex] = inp[n_pos_ex : 2 * n_pos_ex]
    lis_inp[n_pos_ex:] = inp[midp + n_pos_ex : midp + (2 * n_pos_ex)]

    lis_label = torch.zeros(n_examples, dtype=torch.uint8)
    lis_label[:n_pos_ex] = 1

    if percent_novel == 0.0:
        lis_inp = spk_inp
        lis_label = spk_label
    elif percent_novel < 1.0:  # Sample some negatives
        is_novel = torch.rand(n_pos_ex) < percent_novel
        if spk_inp.ndim == 4:  # Image
            is_novel_exp = is_novel.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:  # Feat
            is_novel_exp = is_novel.unsqueeze(1)

        lis_inp = torch.where(is_novel_exp, lis_inp, spk_inp)
        lis_label = torch.where(is_novel, lis_label, spk_label)

    return spk_inp, spk_label, lis_inp, lis_label


def return_index(getitem):
    def with_index(self, index):
        res = getitem(self, index)
        return res + (index,)

    return with_index
