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

from typing import Union, Tuple, List
import numpy as np
import torch
from skimage import measure
try:
    import cubvh
except ImportError:
    cubvh = None

def get_sparse_valid_voxels(grid_logit: torch.Tensor):
    if not isinstance(grid_logit, torch.Tensor):
        raise TypeError('Input must be a PyTorch tensor.')
    if grid_logit.dim() != 3 or grid_logit.shape[0] != grid_logit.shape[1] or grid_logit.shape[0] != grid_logit.shape[2]:
        raise ValueError('Input tensor must have shape (N, N, N)')
    N = grid_logit.shape[0]
    device = grid_logit.device
    chunk_size = 128
    all_sparse_coords = []
    all_sparse_logits = []
    for start_x in range(0, N - 1, chunk_size):
        end_x = min(start_x + chunk_size, N - 1)
        slice_end = end_x + 1
        chunk = grid_logit[start_x:slice_end, :, :]
        nan_mask = torch.isnan(chunk)
        sub_nan_mask = nan_mask
        invalid_voxel_mask = sub_nan_mask[:-1, :-1, :-1] | sub_nan_mask[1:, :-1, :-1] | sub_nan_mask[:-1, 1:, :-1] | sub_nan_mask[:-1, :-1, 1:] | sub_nan_mask[:-1, 1:, 1:] | sub_nan_mask[1:, :-1, 1:] | sub_nan_mask[1:, 1:, :-1] | sub_nan_mask[1:, 1:, 1:]
        valid_voxel_mask = ~invalid_voxel_mask
        local_coords = valid_voxel_mask.nonzero(as_tuple=False)
        if local_coords.shape[0] > 0:
            lx, ly, lz = (local_coords[:, 0], local_coords[:, 1], local_coords[:, 2])
            sparse_vertex_logits = torch.stack([chunk[lx, ly, lz], chunk[lx + 1, ly, lz], chunk[lx + 1, ly + 1, lz], chunk[lx, ly + 1, lz], chunk[lx, ly, lz + 1], chunk[lx + 1, ly, lz + 1], chunk[lx + 1, ly + 1, lz + 1], chunk[lx, ly + 1, lz + 1]], dim=1)
            global_coords = local_coords.clone()
            global_coords[:, 0] += start_x
            all_sparse_coords.append(global_coords)
            all_sparse_logits.append(sparse_vertex_logits)
        del chunk, nan_mask, invalid_voxel_mask, valid_voxel_mask, local_coords
    if not all_sparse_coords:
        return (torch.empty((0, 3), dtype=torch.long, device=device), torch.empty((0, 8), dtype=grid_logit.dtype, device=device))
    sparse_coords = torch.cat(all_sparse_coords, dim=0)
    sparse_vertex_logits = torch.cat(all_sparse_logits, dim=0)
    return (sparse_coords, sparse_vertex_logits)


def extract_mesh(grid, bounds, level):
    import trimesh
    import warnings
    if cubvh is not None and grid.is_cuda:
        coords, logits = get_sparse_valid_voxels(grid)
        if coords.numel() == 0:
            raise RuntimeError('No valid cells for marching cubes')
        vertices, faces = cubvh.sparse_marching_cubes(coords, logits.float(), level)
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
    else:
        # The finite mask excludes unqueried octree cells.
        values = grid.float().cpu().numpy()
        valid = np.isfinite(values)
        if not valid.any() or not (values[valid].min() < level < values[valid].max()):
            raise RuntimeError('Decoded field does not cross the requested surface level')
        warnings.warn('Using scikit-image marching cubes; cubvh matches the original GPU extractor.', stacklevel=2)
        cells = valid[:-1, :-1, :-1].copy()
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    cells &= valid[dx:dx+values.shape[0]-1, dy:dy+values.shape[1]-1, dz:dz+values.shape[2]-1]
        complete = np.zeros_like(valid)
        complete[1:, 1:, 1:] = cells
        vertices, faces, _, _ = measure.marching_cubes(
            np.nan_to_num(values, nan=level + 10000), level=level, mask=complete)
    if len(faces) == 0:
        raise RuntimeError('Marching cubes produced an empty mesh')
    vertices = vertices / (np.array(grid.shape) - 1) * (2 * bounds) - bounds
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
