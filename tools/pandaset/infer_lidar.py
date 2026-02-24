"""
Stage 1 (LiDAR-guided): Run MapTR inference on PandaSet with LiDAR depth correction.

PandaSet provides per-frame LiDAR point clouds.  The nuScenes-trained depth
network is out-of-distribution for PandaSet cameras (different FOV, different
scene appearance), so its per-pixel depth estimates are unreliable.

This script projects each frame's LiDAR point cloud into the 6 camera images
and builds a sparse one-hot depth distribution at feature-map resolution
(1/32 of the padded image).  Wherever LiDAR provides a depth value that depth
bin is used verbatim; network predictions are kept only for sky/pixels the
LiDAR does not cover.

The injection uses  LSSTransform._lidar_depth  (set before each forward call,
cleared after).  The encoder's get_cam_feats() checks this attribute and
substitutes it for the learned depth where the mask is non-zero.

Usage:
    python tools/pandaset/infer_lidar.py \\
        projects/configs/maptrv2/maptrv2_nusc_r50_pandaset.py \\
        ckpts/maptrv2_nusc_r50_24ep.pth \\
        --pred-dir work_dirs/maptrv2_nusc_001_lidar/preds \\
        --ann-file data/pandaset/pandaset_map_infos_seq001_test.pkl
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import projects.mmdet3d_plugin.compat  # noqa

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from scipy.spatial.transform import Rotation as Rscipy

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


# ── LiDAR depth helpers ───────────────────────────────────────────────────────

def _quat_to_rot(q: dict) -> np.ndarray:
    """Convert {'w','x','y','z'} dict → 3×3 rotation matrix."""
    return Rscipy.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()


def compute_lidar_depth_dist(
    pts_world: np.ndarray,      # (M, 3)  LiDAR pts in world/GPS frame
    lidar2img_list,             # list of N (4,4) scaled lidar→image matrices
    feat_down_sample: int,      # 32
    fH: int,                    # padded image H // feat_down_sample
    fW: int,                    # padded image W // feat_down_sample
    dbound,                     # [d_min, d_max, d_step]
) -> np.ndarray:                # (N, D, fH, fW) float32
    """
    Project LiDAR world-frame points into each camera and create a one-hot
    depth-bin distribution at feature-map resolution.

    The returned array has 1.0 at the correct depth bin and 0 elsewhere.
    Pixels with no LiDAR coverage are all-zeros (network depth is kept).
    """
    d_min, d_max, d_step = dbound
    D = int(round((d_max - d_min) / d_step))
    N = len(lidar2img_list)

    depth_dist = np.zeros((N, D, fH, fW), dtype=np.float32)

    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1), dtype=np.float32)])  # (M, 4)

    for cam_idx, l2i in enumerate(lidar2img_list):
        l2i = np.asarray(l2i, dtype=np.float64)
        p = (l2i @ pts_h.T).T   # (M, 4)  — homogeneous image coords

        # Keep points in front of camera and within depth range
        z = p[:, 2]
        valid = (z > d_min) & (z < d_max)
        if not valid.any():
            continue
        z_v  = z[valid]
        u_v  = p[valid, 0] / z_v    # pixel x in scaled image
        v_v  = p[valid, 1] / z_v    # pixel y in scaled image

        # Convert to feature-map coordinates
        u_f = u_v / feat_down_sample
        v_f = v_v / feat_down_sample

        # Filter to feature map bounds
        in_map = (u_f >= 0) & (u_f < fW) & (v_f >= 0) & (v_f < fH)
        if not in_map.any():
            continue
        u_f = u_f[in_map]
        v_f = v_f[in_map]
        z_v = z_v[in_map]

        # Quantize to integer cell and depth bin
        u_i   = np.clip(np.round(u_f).astype(np.int32), 0, fW - 1)
        v_i   = np.clip(np.round(v_f).astype(np.int32), 0, fH - 1)
        d_bin = np.clip(((z_v - d_min) / d_step).astype(np.int32), 0, D - 1)

        # For each feature cell keep only the nearest (minimum depth) point.
        # Sort ascending by depth so that np.unique retains the first = nearest.
        order  = np.argsort(z_v)
        u_i    = u_i[order];  v_i = v_i[order];  d_bin = d_bin[order]
        cells  = v_i * fW + u_i
        _, first = np.unique(cells, return_index=True)

        u_keep = u_i[first];  v_keep = v_i[first];  d_keep = d_bin[first]
        depth_dist[cam_idx, d_keep, v_keep, u_keep] = 1.0

    return depth_dist


def _find_lss_encoder(model):
    """Walk the model tree and return the LSSTransform module (or None)."""
    for m in model.modules():
        if type(m).__name__ == 'LSSTransform':
            return m
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--pred-dir', default=None)
    p.add_argument('--ann-file', default=None)
    p.add_argument('--depthnet-ckpt', default=None,
                   help='Fine-tuned DepthNet state_dict to load before inference')
    p.add_argument('--no-lidar-correction', action='store_true',
                   help='Disable LiDAR depth injection (useful after fine-tuning)')
    return p.parse_args()


def main():
    args     = parse_args()
    cfg_name = osp.splitext(osp.basename(args.config))[0]
    if args.pred_dir is None:
        args.pred_dir = osp.join('work_dirs', cfg_name + '_lidar', 'preds')
    mmcv.mkdir_or_exist(args.pred_dir)

    logger = get_root_logger()
    logger.info(f'LiDAR-guided predictions → {args.pred_dir}')

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin') and cfg.get('plugin_dir'):
        import importlib
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))

    cfg.model.pretrained = None
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        samples_per_gpu = 1

    dataset = build_dataset(cfg.data.test)
    nonshuffler_sampler = cfg.data.get('nonshuffler_sampler',
                                        dict(type='DistributedSampler'))
    data_loader = build_dataloader(dataset, samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=0, dist=False, shuffle=False,
                                   nonshuffler_sampler=nonshuffler_sampler)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg:
        wrap_fp16_model(model)

    _orig_torch_load = torch.load
    def _torch_load_compat(f, *a, **kw):
        kw.setdefault('weights_only', False)
        return _orig_torch_load(f, *a, **kw)
    torch.load = _torch_load_compat
    try:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    finally:
        torch.load = _orig_torch_load

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Locate the LSSTransform encoder for depth injection
    lss = _find_lss_encoder(model)
    if lss is None:
        logger.warning('LSSTransform not found — running without LiDAR depth correction')
    else:
        logger.info('LSSTransform found — LiDAR depth injection enabled')

    # Optionally load a fine-tuned DepthNet
    if args.depthnet_ckpt and lss is not None:
        state = torch.load(args.depthnet_ckpt, map_location='cpu',
                           weights_only=False)
        lss.depth_net.load_state_dict(state)
        logger.info(f'Loaded fine-tuned DepthNet from {args.depthnet_ckpt}')

    # Depth-bin config from model config
    dbound          = cfg.dbound           # [d_min, d_max, d_step]
    feat_down_sample = 32                  # hard-coded in all nuScenes configs
    pc_range        = list(cfg.point_cloud_range)

    prog_bar = mmcv.ProgressBar(len(dataset))

    for frame_idx, data in enumerate(data_loader):
        img_metas    = data['img_metas'][0].data[0]
        lidar2global = np.array(img_metas['lidar2global'], dtype=np.float64)
        # 'lidar_path' is not in meta_keys; fall back to 'pts_filename'
        lidar_path   = img_metas.get('lidar_path', '') or img_metas.get('pts_filename', '')

        # ── LiDAR depth injection ──────────────────────────────────────────
        if lss is not None and not args.no_lidar_correction \
                and lidar_path and osp.isfile(lidar_path):
            try:
                # Load PandaSet point cloud (world/GPS frame, pandas DataFrame)
                pts_df    = pd.read_pickle(lidar_path)
                pts_world = pts_df[['x', 'y', 'z']].values.astype(np.float32)

                # scaled lidar2img from img_metas (already has scale applied)
                lidar2img_list = img_metas['lidar2img']   # list of N (4,4) arrays

                # Feature map size from the actual image tensor.
                # Shape can be (B, N, C, H, W) or (N, C, H, W) depending on batch wrapping.
                img_tensor = data['img'][0].data[0]
                img_H, img_W = img_tensor.shape[-2], img_tensor.shape[-1]
                fH = img_H // feat_down_sample
                fW = img_W // feat_down_sample
                N  = len(lidar2img_list)

                # Build one-hot depth distribution  (N, D, fH, fW)
                depth_dist = compute_lidar_depth_dist(
                    pts_world, lidar2img_list,
                    feat_down_sample, fH, fW, dbound,
                )

                # Set on LSSTransform as (B*N, D, fH, fW) = (N, D, fH, fW) for B=1
                lss._lidar_depth = torch.from_numpy(depth_dist).cuda()
            except Exception as e:
                logger.warning(f'Frame {frame_idx}: LiDAR depth failed ({e}), using network depth')
                lss._lidar_depth = None
        elif lss is not None:
            lss._lidar_depth = None

        # ── Model inference ────────────────────────────────────────────────
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # Clear cached LiDAR depth
        if lss is not None:
            lss._lidar_depth = None

        res       = result[0]['pts_bbox']
        scores    = res['scores_3d'].numpy()
        labels    = res['labels_3d'].numpy().astype(np.int32)
        pts_local = res['pts_3d'].numpy()[..., :2]
        filenames  = img_metas.get('filename', [])

        record = {
            'ego_xy':       lidar2global[:2, 3].tolist(),
            'lidar2global': lidar2global.tolist(),
            'scores':       scores.tolist(),
            'labels':       labels.tolist(),
            'pts_local':    pts_local.tolist(),
            'filenames':    list(filenames),
            'pc_range':     pc_range,
            'scene_token':  img_metas.get('scene_token', ''),
            'lidar_path':   lidar_path,
        }
        with open(osp.join(args.pred_dir, f'frame_{frame_idx:04d}.json'), 'w') as f:
            json.dump(record, f)
        prog_bar.update()

    logger.info(f'\nDone — {len(dataset)} frames saved to {args.pred_dir}')


if __name__ == '__main__':
    main()
