"""Build voxel and image conditions from a segmented scaffold."""
import cv2
import numpy as np
import torch
import trimesh
from torchvision import transforms


def _fps(voxels, count):
    if len(voxels) <= count:
        return voxels
    pts = voxels.astype(np.float32)
    dist = np.full(len(pts), np.inf, dtype=np.float32)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = np.argmin(((pts - pts.mean(0, keepdims=True)) ** 2).sum(1))
    for i in range(1, count):
        np.minimum(dist, ((pts - pts[chosen[i - 1]]) ** 2).sum(1), out=dist)
        chosen[i] = np.argmax(dist)
    return voxels[chosen]


def _voxelize(points, resolution):
    idx = np.clip(np.floor((points + 1) * 0.5 * resolution).astype(np.int64), 0, resolution - 1)
    flat = idx[:, 0] * resolution ** 2 + idx[:, 1] * resolution + idx[:, 2]
    _, first = np.unique(flat, return_index=True)
    return idx[np.sort(first)]


def prepare_geometry(mesh_path, labels_path, config, seed):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 1 or len(labels) != len(mesh.faces) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must contain one integer part ID per mesh face')
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if not len(mesh.faces) or not np.isfinite(verts).all():
        raise ValueError('Input mesh must have faces and finite vertices')
    center = (verts.min(0) + verts.max(0)) * 0.5
    extent = float(np.ptp(verts, axis=0).max())
    if extent <= 1e-6:
        raise ValueError('Input mesh has zero extent')
    shape_scale = 1.96 / extent
    verts = (verts - center) * shape_scale
    part_ids = sorted(np.unique(labels).tolist())
    rng = np.random.default_rng(seed)
    voxel_list, bboxes, global_bboxes = [], [], []
    for part_id in part_ids:
        part = trimesh.Trimesh(vertices=verts, faces=mesh.faces[labels == part_id], process=False)
        points, _ = trimesh.sample.sample_surface(part, config['surface_points'], seed=int(rng.integers(0, 2**31)))
        points = np.asarray(points, dtype=np.float32)
        if not np.isfinite(points).all():
            raise ValueError(f'Part {part_id} has an invalid surface')
        mn, mx = points.min(0), points.max(0)
        ctr = (mn + mx) * 0.5
        scale = float(np.maximum(mx - mn, 1e-6).max()) * 0.5 / config['part_unit_scale']
        voxels = _fps(_voxelize((points - ctr) / scale, config['voxel_resolution']), config['max_voxels'])
        voxel_list.append(torch.from_numpy(voxels))
        bboxes.append(np.r_[ctr, np.float32(scale)])
        global_bboxes.append(np.stack([mn, mx]))
    max_tokens = max(map(len, voxel_list))
    voxel_cond = torch.zeros(len(part_ids), max_tokens, 3, dtype=torch.long)
    padding_mask = torch.zeros(len(part_ids), max_tokens, dtype=torch.bool)
    for i, voxels in enumerate(voxel_list):
        voxel_cond[i, :len(voxels)] = voxels
        padding_mask[i, :len(voxels)] = True
    return {'voxel_cond':voxel_cond, 'padding_mask':padding_mask,
            'part_bbox':torch.from_numpy(np.asarray(bboxes, dtype=np.float32)),
            'part_global_bbox_minmax':torch.from_numpy(np.asarray(global_bboxes, dtype=np.float32)),
            'part_ids':part_ids, 'shape_center':center, 'shape_scale':shape_scale}


def prepare_image(image_path, mask_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise FileNotFoundError('Could not read the input image or foreground mask')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        image = (cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    else:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float32)
    if mask.max() <= 1 + 1e-6:
        mask *= 255
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    foreground = np.argwhere(mask > 0.3 * 255)
    if not len(foreground):
        raise ValueError('Foreground mask is empty; provide a mask with foreground > 76.5')
    lo, hi = foreground.min(0), foreground.max(0)
    image = image[max(lo[0]-5,0):min(hi[0]+5,h), max(lo[1]-5,0):min(hi[1]+5,w)]
    h, w = image.shape[:2]
    side = int(max(h, w) * config['padding_ratio'])
    padded = np.full((side, side, 3), 255, dtype=np.uint8)
    y, x = (side-h)//2, (side-w)//2
    padded[y:y+h, x:x+w] = image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(config['image_size']),
                                    transforms.Normalize([0.5]*3, [0.5]*3)])
    return transform(padded).unsqueeze(0)
