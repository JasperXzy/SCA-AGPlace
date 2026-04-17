# ruff: noqa: NPY002

import copy

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as T
from PIL import Image

# Utonia official input pipeline constants (transform.default(scale=0.2)
# + GridSample(grid_size=0.01)). Effective voxel = 0.01 / 0.2 = 0.05 m.
UTONIA_GLOBAL_SCALE = 0.2
UTONIA_GRID_SIZE = 0.01


def estimate_normals_o3d(pts, k=30):
    """Per-point normal estimation via Open3D, matching the Utonia paper:
        "Surface normals are estimated with Open3D, including directions
         from points to the LiDAR center."

    Args:
        pts: np.ndarray [N, 3] (velodyne XYZ)
        k: KNN neighborhood size (paper uses 30)
    Returns:
        normals: np.ndarray [N, 3] float32, oriented toward the LiDAR origin (0,0,0)
    """
    n = pts.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(k, n)))
    # Critical: orient every normal to point TOWARD the LiDAR sensor (origin in
    # velodyne frame). Without this, PCA-derived normals have random sign and
    # become a high-frequency noise channel for the pretrained Utonia encoder.
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def _rand_rotation_outdoor_scene():
    """Full yaw (z) rotation + mild roll/pitch perturbations, matching
    Utonia outdoor scene-level augmentation (Sec 4.1):
        z in [-pi, pi], x/y in [-pi/16, pi/16].
    """
    theta_z = 2.0 * np.pi * (np.random.rand() - 0.5)  # [-pi, pi]
    theta_x = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)  # [-pi/16, pi/16]
    theta_y = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _rand_scale_jitter(eta=1.1):
    """Isotropic multiplicative rescale r ~ exp(U(-log eta, log eta)).
    Matches Sec A.2 / Sec 4.1: ±10% scale jitter for scene-level data.
    """
    log_eta = np.log(eta)
    return float(np.exp(np.random.uniform(-log_eta, log_eta)))


def _rand_anisotropic_jitter(eta=1.1):
    """Anisotropic per-axis jitter j ~ exp(U(-log eta, log eta)^3), matching
    DINOv3-style coordinate augmentation used by Utonia (Sec 3.3, Sec A.2).
    """
    log_eta = np.log(eta)
    return np.exp(np.random.uniform(-log_eta, log_eta, size=3)).astype(np.float32)


def _causal_modality_blind(rgb, normal, p_drop_rgb=0.3, p_drop_normal=0.3):
    """Per-sample (per-data) causal modality blinding (Sec 3.1).
    Randomly zero out the entire RGB or Normal modality for a sample, so the
    model does not over-rely on any single modality. Matches the
    with-color / without-color partitioning in Tab. 7(d).
    """
    if np.random.rand() < p_drop_rgb:
        rgb = np.zeros_like(rgb)
    if np.random.rand() < p_drop_normal:
        normal = np.zeros_like(normal)
    return rgb, normal


def _dedup_grid(coords_int_np):
    """Return indices to keep (order-preserving) so that int grid coords are unique."""
    _, keep = np.unique(coords_int_np, axis=0, return_index=True)
    keep.sort()
    return keep


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


def load_qimage(datapath, split, args):
    image = Image.open(datapath)
    image = image.convert("RGB")
    if split == "train":
        tf = T.Compose(
            [
                T.Resize((args.q_resize, args.q_resize)),
                T.ColorJitter(
                    brightness=args.q_jitter,
                    contrast=args.q_jitter,
                    saturation=args.q_jitter,
                    hue=min(0.5, args.q_jitter),
                ),
                T.ToTensor(),
                #   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                T.Normalize(mean=0.5, std=0.22),
            ]
        )
    elif split == "test":
        tf = T.Compose(
            [
                T.Resize((args.q_resize, args.q_resize)),
                T.ToTensor(),
                #   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                T.Normalize(mean=0.5, std=0.22),
            ]
        )
    image = tf(image)
    return image


def load_dbimage(datapath, split, args):
    image = Image.open(datapath)
    image = image.convert("RGB")
    if split == "train":
        tf = T.Compose(
            [
                T.CenterCrop(args.db_cropsize),
                T.Resize((args.db_resize, args.db_resize)),
                T.ColorJitter(
                    brightness=args.db_jitter,
                    contrast=args.db_jitter,
                    saturation=args.db_jitter,
                    hue=min(0.5, args.db_jitter),
                ),
                T.ToTensor(),
                #   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                T.Normalize(mean=0.5, std=0.22),
            ]
        )
    elif split == "test":
        tf = T.Compose(
            [
                T.CenterCrop(args.db_cropsize),
                T.Resize((args.db_resize, args.db_resize)),
                T.ToTensor(),
                #   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                T.Normalize(mean=0.5, std=0.22),
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


def load_pc_sph_bev(file_path, split):  # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)  # for kitti360_voxel
    sph = torch.empty(0)
    bev = torch.empty(0)
    return pc, sph, bev
