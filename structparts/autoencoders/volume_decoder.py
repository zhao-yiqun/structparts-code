# Modified for the StructParts inference-only release: removed unused paths and dependencies.
# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.
# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List, Callable, Optional
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm
logger = logging.getLogger(__name__)

def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    val = input_tensor + alpha
    valid_mask = val > -9000
    mask = torch.ones_like(val, dtype=torch.int32)
    sign = torch.sign(val.to(torch.float32))

    def check_neighbor_sign(shift, axis):
        if shift == 0:
            return
        pad_dims = [0, 0, 0, 0, 0, 0]
        if axis == 0:
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)
        padded = F.pad(val.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')
        slice_dims = [slice(None)] * 3
        if axis == 0:
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)
        padded = padded.squeeze(0).squeeze(0)
        neighbor = padded[slice_dims]
        neighbor = torch.where(neighbor > -9000, neighbor, val)
        neighbor_sign = torch.sign(neighbor.to(torch.float32))
        return neighbor_sign == sign
    directions = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)]
    for shift, axis in directions:
        is_same = check_neighbor_sign(shift, axis)
        mask = mask & is_same.to(torch.int32)
    mask = (~mask.bool()).to(torch.int32)
    return mask * valid_mask.to(torch.int32)

def generate_dense_grid_points(bbox_min: np.ndarray, bbox_max: np.ndarray, octree_resolution: int, indexing: str='ij'):
    length = bbox_max - bbox_min
    num_cells = octree_resolution
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]
    return (xyz, grid_size, length)

class CoarseToFineVolumeDecoding:

    @torch.no_grad()
    def __call__(self, latents: torch.FloatTensor, geo_decoder: Callable, bounds: Union[Tuple[float], List[float], float]=1.01, num_chunks: int=10000, mc_level: float=-1.0 / 512, octree_resolution: int=None, min_resolution: int=95, enable_pbar: bool=True, part_bbox: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, bbox_sdf_value: float=2.0, bbox_at_final: bool=False, **kwargs):
        if octree_resolution < 2 or octree_resolution & (octree_resolution - 1):
            raise ValueError("resolution must be a power of two >= 2")
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]
        resolutions = []
        res = octree_resolution
        if res < min_resolution:
            resolutions.append(res)
        while res >= min_resolution:
            resolutions.append(res)
            res = res // 2
        resolutions.reverse()
        logger.info('CoarseToFineVolumeDecoding resolutions: %s', resolutions)
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))
        xyz_samples, grid_size, _length = generate_dense_grid_points(bbox_min=bbox_min, bbox_max=bbox_max, octree_resolution=resolutions[0], indexing='ij')
        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)
        if part_bbox is not None:
            pb_min, pb_max = part_bbox
            inside_mask = ((xyz_samples >= pb_min) & (xyz_samples <= pb_max)).all(dim=-1)
            inside_indices = inside_mask.nonzero(as_tuple=True)[0]
            xyz_inside = xyz_samples[inside_indices]
            total_pts = xyz_samples.shape[0]
            logger.info('CoarseToFine bbox crop (coarse): %d / %d inside (%.1f%% skipped)', xyz_inside.shape[0], total_pts, 100.0 * (1 - xyz_inside.shape[0] / total_pts))
            all_logits = torch.full((batch_size, total_pts, 1), bbox_sdf_value, device=device, dtype=dtype)
            coarse_logits = []
            for start in tqdm(range(0, xyz_inside.shape[0], num_chunks), desc=f'CoarseToFine [r{resolutions[0] + 1}] bbox', disable=not enable_pbar):
                q = xyz_inside[start:start + num_chunks]
                q = repeat(q, 'p c -> b p c', b=batch_size)
                coarse_logits.append(geo_decoder(queries=q, latents=latents))
            if coarse_logits:
                coarse_logits = torch.cat(coarse_logits, dim=1)
                all_logits[:, inside_indices, :] = coarse_logits
            grid_logits = all_logits.view((batch_size, *grid_size)).float()
        else:
            batch_logits = []
            for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f'CoarseToFine [r{resolutions[0] + 1}]', disable=not enable_pbar):
                q = xyz_samples[start:start + num_chunks]
                q = repeat(q, 'p c -> b p c', b=batch_size)
                batch_logits.append(geo_decoder(queries=q, latents=latents))
            grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, *grid_size)).float()
        for depth_idx, octree_depth_now in enumerate(resolutions[1:]):
            is_final = octree_depth_now == resolutions[-1]
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            abs_thresh = min(0.95, 0.95 * 128 * 4 / octree_depth_now)
            curr_points += grid_logits.squeeze(0).abs() < abs_thresh
            if octree_depth_now > 510:
                expand_num = 0
            else:
                expand_num = 1
            for _ in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            cidx_x, cidx_y, cidx_z = torch.where(curr_points > 0)
            if cidx_x.shape[0] == 0:
                logger.warning('CoarseToFine: no near-surface voxels at resolution %d; stopping (using coarser grid).', octree_depth_now + 1)
                break
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for _ in range(1):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)
            next_index_shape = next_index.shape
            del next_index
            torch.cuda.empty_cache()
            next_points = torch.stack(nidx, dim=1)
            next_points = next_points * torch.tensor(resolution, dtype=torch.float32, device=device) + torch.tensor(bbox_min, dtype=torch.float32, device=device)
            if is_final and bbox_at_final and (part_bbox is not None):
                pb_min_t = part_bbox[0].to(device=device, dtype=torch.float32).view(1, 3)
                pb_max_t = part_bbox[1].to(device=device, dtype=torch.float32).view(1, 3)
                keep = ((next_points >= pb_min_t) & (next_points <= pb_max_t)).all(dim=-1)
                n_in = int(keep.sum().item())
                logger.info('CoarseToFine bbox crop (final r%d): %d / %d queries kept (%.1f%% dropped)', octree_depth_now + 1, n_in, next_points.shape[0], 100.0 * (1.0 - n_in / max(next_points.shape[0], 1)))
                next_points = next_points[keep]
                nidx = tuple((n[keep] for n in nidx))
            if next_points.shape[0] == 0:
                logger.warning('CoarseToFine: empty next_points at resolution %d; stopping.', octree_depth_now + 1)
                break
            logger.info('CoarseToFine [r%d]: %d query points (%.1f%% of full grid)', octree_depth_now + 1, next_points.shape[0], 100.0 * next_points.shape[0] / np.prod(grid_size))
            batch_logits = []
            for start in tqdm(range(0, next_points.shape[0], num_chunks), desc=f'CoarseToFine [r{octree_depth_now + 1}]', disable=not enable_pbar):
                q = next_points[start:start + num_chunks]
                q = repeat(q, 'p c -> b p c', b=batch_size).to(latents.dtype)
                batch_logits.append(geo_decoder(queries=q, latents=latents))
            if is_final:
                next_logits = torch.full(next_index_shape, float('nan'), dtype=dtype, device=device)
            else:
                next_logits = torch.full(next_index_shape, -10000.0, dtype=dtype, device=device)
            grid_logits = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)
        grid_logits[grid_logits == -10000.0] = float('nan')
        return grid_logits
