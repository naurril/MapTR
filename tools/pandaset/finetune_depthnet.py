"""
Fine-tune LSSTransform DepthNet on PandaSet using LiDAR depth supervision.

All model parameters are frozen except the DepthNet inside LSSTransform.
LiDAR point clouds are projected into each camera to produce sparse one-hot
depth labels at feature-map resolution.  BCE loss is computed only at pixels
that have LiDAR coverage; everything else is ignored.

The fine-tuned DepthNet weights are saved every epoch so they can be loaded
into the regular inference scripts via --depthnet-ckpt.

Usage:
    python tools/pandaset/finetune_depthnet.py \\
        projects/configs/maptrv2/maptrv2_nusc_r50_pandaset.py \\
        ckpts/maptrv2_nusc_r50_24ep.pth \\
        --ann-files data/pandaset/pandaset_map_infos_001.pkl \\
                    data/pandaset/pandaset_map_infos_008.pkl \\
                    data/pandaset/pandaset_map_infos_040.pkl \\
        --save-dir  work_dirs/depthnet_pandaset \\
        --epochs 10 --lr 1e-4
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import projects.mmdet3d_plugin.compat  # noqa

import argparse
import os.path as osp

import mmcv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# Reuse LiDAR projection helper from infer_lidar
from infer_lidar import compute_lidar_depth_dist, _find_lss_encoder


# ── helpers ───────────────────────────────────────────────────────────────────

def _freeze_all_except_depthnet(model, lss):
    """Freeze everything; unfreeze lss.depth_net; put depth_net in train mode."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lss.depth_net.parameters():
        p.requires_grad_(True)
    model.eval()           # BN in frozen layers stays in eval (no stat updates)
    lss.depth_net.train()  # BN in depth_net updates its running stats


MIN_RANGE = 8.0   # metres — ignore LiDAR returns closer than this


def _depth_bce_loss(depth_pred, lidar_gt, dbound):
    """
    Range-filtered, depth-weighted BCE loss.

    Near-range LiDAR returns (< MIN_RANGE) are excluded: they correspond to
    the vehicle's immediate surroundings and would bias the network toward
    predicting very short depths, collapsing BEV features around the vehicle.
    Among the remaining pixels each one is weighted by its normalised depth
    (d / d_max) so that far-range supervision is not drowned out.

    Args:
        depth_pred (N, D, fH, fW): softmax depth from get_cam_feats.
        lidar_gt   (N, D, fH, fW): one-hot LiDAR depth (0 = no coverage).
        dbound     [d_min, d_max, d_step]: depth bin config.
    Returns:
        scalar loss tensor (or 0.0 if no qualifying pixels).
    """
    d_min, d_max, d_step = dbound
    D = depth_pred.shape[1]

    # Expected LiDAR depth per pixel: (N, fH, fW)
    bin_vals = (torch.arange(D, device=lidar_gt.device, dtype=lidar_gt.dtype)
                * d_step + d_min)
    d_lidar  = (lidar_gt * bin_vals.view(1, D, 1, 1)).sum(1)   # (N, fH, fW)

    # Only supervise pixels with LiDAR coverage beyond MIN_RANGE
    fg_mask = (lidar_gt.sum(1) > 0.5) & (d_lidar > MIN_RANGE)  # (N, fH, fW)
    n_fg = fg_mask.sum().item()
    if n_fg == 0:
        return depth_pred.new_tensor(0.0)

    # Per-pixel weight: farther returns count more
    weight = (d_lidar / d_max).clamp(min=0.2)                   # (N, fH, fW)

    pred_flat   = depth_pred.permute(0, 2, 3, 1).reshape(-1, D)
    gt_flat     = lidar_gt.permute(0, 2, 3, 1).reshape(-1, D)
    weight_flat = weight.reshape(-1)
    fg_flat     = fg_mask.reshape(-1)

    bce = F.binary_cross_entropy(
        pred_flat[fg_flat].clamp(1e-6, 1 - 1e-6),
        gt_flat[fg_flat],
        reduction='none',
    ).sum(1)                                                     # (n_fg,)

    loss = (bce * weight_flat[fg_flat]).sum() / weight_flat[fg_flat].sum()
    return loss


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-files', nargs='+', required=True,
                   help='One or more PandaSet pkl ann-files for training')
    p.add_argument('--save-dir', default=None)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lidar-pixels', type=int, default=30,
                   help='Skip frames with fewer LiDAR feature-map pixels than this')
    p.add_argument('--depthnet-ckpt', default=None,
                   help='Resume from a previously saved depthnet checkpoint')
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg_name = osp.splitext(osp.basename(args.config))[0]
    if args.save_dir is None:
        args.save_dir = osp.join('work_dirs', cfg_name + '_depthnet')
    mmcv.mkdir_or_exist(args.save_dir)

    logger = get_root_logger()
    logger.info(f'DepthNet fine-tune → {args.save_dir}')

    # ── config & model ────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin') and cfg.get('plugin_dir'):
        import importlib
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))

    cfg.model.pretrained = None
    cfg.model.train_cfg  = None

    model   = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
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

    lss = _find_lss_encoder(model)
    if lss is None:
        logger.error('LSSTransform not found — cannot fine-tune DepthNet')
        return
    logger.info('LSSTransform found')

    # Optionally resume depth_net from a previous run
    if args.depthnet_ckpt:
        state = torch.load(args.depthnet_ckpt, map_location='cpu',
                           weights_only=False)
        lss.depth_net.load_state_dict(state)
        logger.info(f'Resumed depth_net from {args.depthnet_ckpt}')

    _freeze_all_except_depthnet(model, lss)
    n_params = sum(p.numel() for p in lss.depth_net.parameters())
    logger.info(f'Trainable DepthNet params: {n_params:,}')

    optimizer = torch.optim.Adam(lss.depth_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Depth-bin config
    dbound          = cfg.dbound
    feat_down_sample = 32
    D = int(round((dbound[1] - dbound[0]) / dbound[2]))

    # ── build dataset(s) ──────────────────────────────────────────────────────
    # We may have multiple ann-files (sequences); build one dataset per file
    # and interleave frames across all of them via ConcatDataset.
    from torch.utils.data import ConcatDataset

    datasets = []
    for ann_file in args.ann_files:
        cfg_copy = cfg.copy()
        if isinstance(cfg_copy.data.test, dict):
            cfg_copy.data.test.test_mode = True
            cfg_copy.data.test.ann_file  = ann_file
            spg = cfg_copy.data.test.pop('samples_per_gpu', 1)
            if spg > 1:
                cfg_copy.data.test.pipeline = replace_ImageToTensor(
                    cfg_copy.data.test.pipeline)
        ds = build_dataset(cfg_copy.data.test)
        datasets.append(ds)
        logger.info(f'  {ann_file}: {len(ds)} frames')

    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    nonshuffler_sampler = cfg.data.get('nonshuffler_sampler',
                                       dict(type='DistributedSampler'))
    # Use shuffle=True so the model sees frames in random order each epoch.
    # workers=0 keeps it simple (no multiprocessing issues with DataContainers).
    data_loader = build_dataloader(
        full_dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
        nonshuffler_sampler=nonshuffler_sampler)

    logger.info(f'Total training frames: {len(full_dataset)}')

    # ── training loop ─────────────────────────────────────────────────────────
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_loss  = 0.0
        epoch_steps = 0
        skipped     = 0

        prog_bar = mmcv.ProgressBar(len(full_dataset))

        for frame_idx, data in enumerate(data_loader):
            img_metas = data['img_metas'][0].data[0]
            lidar_path = (img_metas.get('lidar_path', '')
                          or img_metas.get('pts_filename', ''))

            if not lidar_path or not osp.isfile(lidar_path):
                skipped += 1
                prog_bar.update()
                continue

            # ── LiDAR depth labels ────────────────────────────────────────
            try:
                pts_df    = pd.read_pickle(lidar_path)
                pts_world = pts_df[['x', 'y', 'z']].values.astype(np.float32)
            except Exception as e:
                logger.warning(f'Frame {frame_idx}: LiDAR load failed ({e})')
                skipped += 1
                prog_bar.update()
                continue

            lidar2img_list = img_metas['lidar2img']
            img_tensor     = data['img'][0].data[0]
            img_H, img_W   = img_tensor.shape[-2:]
            fH = img_H // feat_down_sample
            fW = img_W // feat_down_sample
            N  = len(lidar2img_list)

            depth_dist = compute_lidar_depth_dist(
                pts_world, lidar2img_list, feat_down_sample, fH, fW, dbound)
            lidar_gt = torch.from_numpy(depth_dist).cuda()  # (N, D, fH, fW)

            n_fg = int((lidar_gt.sum(1) > 0.5).sum().item())
            if n_fg < args.min_lidar_pixels:
                skipped += 1
                prog_bar.update()
                continue

            # ── forward: backbone+neck (frozen) ───────────────────────────
            # extract_img_feat needs (B, N, C, H, W) — add batch dim if missing.
            img = img_tensor.cuda()
            if img.dim() == 4:
                img = img.unsqueeze(0)   # (N, C, H, W) → (1, N, C, H, W)
            with torch.no_grad():
                img_feats = model.module.extract_img_feat(img, img_metas)
                # img_feats[0]: (B, N, C, fH, fW)  — single FPN level

            # ── forward: depth_net (grad enabled) ─────────────────────────
            optimizer.zero_grad()

            # Build mlp_input from img_metas (same logic as LSSTransform.forward)
            camera2ego        = np.asarray([img_metas['camera2ego']])
            camera_intrinsics = np.asarray([img_metas['camera_intrinsics']])
            img_aug_matrix    = np.asarray([img_metas['img_aug_matrix']])

            camera2ego_t        = img_feats[0].new_tensor(camera2ego)        # (1, N, 4, 4)
            camera_intrinsics_t = img_feats[0].new_tensor(camera_intrinsics) # (1, N, 4, 4)
            img_aug_matrix_t    = img_feats[0].new_tensor(img_aug_matrix)    # (1, N, 4, 4)

            post_rots  = img_aug_matrix_t[..., :3, :3]
            post_trans = img_aug_matrix_t[..., :3, 3]
            mlp_input  = lss.get_mlp_input(
                camera2ego_t, camera_intrinsics_t, post_rots, post_trans)

            # get_cam_feats expects (B, N, C, fH, fW)
            _, depth = lss.get_cam_feats(img_feats[0], mlp_input)
            # depth: (B, N, D, fH, fW); B=1
            depth_pred = depth[0]                 # (N, D, fH, fW)

            loss = _depth_bce_loss(depth_pred, lidar_gt, dbound)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lss.depth_net.parameters(), 35.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            prog_bar.update()

        scheduler.step()

        avg_loss = epoch_loss / max(1, epoch_steps)
        logger.info(
            f'\nEpoch {epoch}/{args.epochs}  '
            f'loss={avg_loss:.4f}  steps={epoch_steps}  skipped={skipped}  '
            f'lr={scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint every epoch
        ckpt_path = osp.join(args.save_dir, f'depthnet_ep{epoch:02d}.pth')
        torch.save(lss.depth_net.state_dict(), ckpt_path)
        logger.info(f'  Saved → {ckpt_path}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = osp.join(args.save_dir, 'depthnet_best.pth')
            torch.save(lss.depth_net.state_dict(), best_path)
            logger.info(f'  Best  → {best_path}')

    logger.info(f'Done.  Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
