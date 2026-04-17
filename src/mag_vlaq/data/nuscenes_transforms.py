import copy
import random

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def path_to_pil_img(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image


def load_dbimage(datapath, split, args):
    image = Image.open(datapath)
    image = image.convert("RGB")
    if split == "train":
        tf = T.Compose(
            [
                # T.CenterCrop(args.db_cropsize),
                T.Resize((args.db_resize, args.db_resize), antialias=True),
                # T.ColorJitter(brightness=args.db_jitter, contrast=args.db_jitter, saturation=args.db_jitter, hue=min(0.5, args.db_jitter)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # T.Normalize(mean=0.5, std=0.22)
            ]
        )
    elif split == "test":
        tf = T.Compose(
            [
                # T.CenterCrop(args.db_cropsize),
                T.Resize((args.db_resize, args.db_resize), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # T.Normalize(mean=0.5, std=0.22)
            ]
        )
    else:
        raise NotImplementedError
    image = tf(image)
    return image


def generate_bev_from_pc(pc, w=200, max_thd=100):
    """
    w+1: bev width
    max_thd: max threshold of x,y,z
    """
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # remove pc outside max_thd
    pc = pc[np.max(np.abs(pc), axis=1) < max_thd]
    bin_max = np.max(pc, axis=0)
    bin_min = np.min(pc, axis=0)
    assert np.all(bin_max <= max_thd)
    assert np.all(bin_min >= -max_thd)
    pc = pc + max_thd
    pc = pc / (2 * max_thd) * w
    pc = pc.astype(np.int64)
    bin_max = np.max(pc, axis=0)
    bev = np.zeros([w + 1, w + 1], dtype=np.float32)
    assert np.all(bin_max <= bev.shape[0])
    bev[pc[:, 0], pc[:, 1]] = pc[:, 2]
    return bev


def generate_sph_from_pc(pc, w=361, h=61, args=None):
    # kitti   w=361  h=61
    # ithaca  w=361  h=101
    # kitti360 w=361 h=61
    # w = 361
    # h = 61
    # if 'ithaca365' in args.dataset_name:
    #     w = 361
    #     h = 101

    # generate spherical projection from pc
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # u-v : h-w
    # u:
    u = np.arctan2(pc[:, 2], np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2))
    u = u / np.pi * 180
    u = u + 25
    u = u * 2
    u = h - u
    # v: [0, 360]
    v = np.arctan2(pc[:, 0], pc[:, 1])
    v = v / np.pi * 180
    v = v + 180
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    uv = np.stack([u, v], axis=1)
    uv = np.array(uv, dtype=np.int32)
    # plt.scatter(uv[:,1], uv[:,0], s=1, c=r, cmap='jet')
    # plt.show()
    if args is not None and "ithaca365" in args.dataset_name:
        ids_h = (uv[:, 0] < h) & (uv[:, 0] >= 0)
        uv = uv[ids_h]
        r = r[ids_h]
    sph = np.zeros([h, w])
    sph[uv[:, 0], uv[:, 1]] = r
    # if 'ithaca365' in args.dataset_name:
    #     sph = sph[25:80]
    # plt.imshow(sph)
    # plt.imsave('sph.png', sph)
    # plt.close()
    # plt.show()
    return sph


def load_pc_bev(file_path, split, args):  # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)  # for kitti360_voxel
    bev = generate_bev_from_pc(pc, w=200, max_thd=100)
    # ==== bev
    bev = Image.fromarray(bev).convert("RGB")
    # bev.show()
    (_w, _h) = bev.size
    assert args.bev_resize <= 1
    resize_ratio = random.uniform(args.bev_resize, 2 - args.bev_resize)
    resize_size = int(resize_ratio * min(_w, _h))
    assert resize_size >= args.bev_cropsize

    if args.bev_resize_mode == "nearest":
        resize_mode = InterpolationMode.NEAREST
    elif args.bev_resize_mode == "bilinear":
        resize_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError

    if args.bev_rotate_mode == "nearest":
        rotate_mode = InterpolationMode.NEAREST
    elif args.bev_rotate_mode == "bilinear":
        rotate_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError

    if split == "train":
        tf = T.Compose(
            [
                T.Resize(resize_size, interpolation=resize_mode, antialias=True),
                T.RandomRotation(args.bev_rotate, interpolation=rotate_mode),
                T.CenterCrop(args.bev_cropsize),
                T.ColorJitter(
                    brightness=args.bev_jitter,
                    contrast=args.bev_jitter,
                    saturation=args.bev_jitter,
                    hue=min(0.5, args.bev_jitter),
                ),
                T.ToTensor(),
                T.Normalize(mean=args.bev_mean, std=args.bev_std),
            ]
        )
    elif split == "test":
        tf = T.Compose(
            [
                T.Resize(resize_size, interpolation=resize_mode, antialias=True),
                T.CenterCrop(args.bev_cropsize),
                T.ToTensor(),
                T.Normalize(mean=args.bev_mean, std=args.bev_std),
            ]
        )
    else:
        raise NotImplementedError
    bev = tf(bev)
    # # ---- DEBUG:viz
    # bev = bev*args.bev_std + args.bev_mean
    # bev = T.ToPILImage()(bev)
    # bev.show()
    return pc, bev


def load_pc_sph(file_path, split, args):  # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)  # for kitti360_voxel
    sph = generate_sph_from_pc(pc, w=361, h=61, args=args)
    # ==== sph
    sph = Image.fromarray(sph).convert("RGB")
    (_w, _h) = sph.size
    assert args.sph_resize <= 1
    resize_ratio = random.uniform(args.sph_resize, 2 - args.sph_resize)
    resize_size = int(resize_ratio * min(_w, _h))
    if split == "train":
        tf = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
                T.ColorJitter(
                    brightness=args.sph_jitter,
                    contrast=args.sph_jitter,
                    saturation=args.sph_jitter,
                    hue=min(0.5, args.sph_jitter),
                ),
                T.ToTensor(),
                T.Normalize(mean=args.sph_mean, std=args.sph_std),
            ]
        )
    elif split == "test":
        tf = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=args.sph_mean, std=args.sph_std),
            ]
        )
    else:
        raise NotImplementedError
    sph = tf(sph)
    return pc, sph
