"""
Stage 1: Run MapTR inference on PandaSet and save per-frame predictions.

Each frame is saved as frame_NNNN.json containing:
  ego_xy       [x, y]                   world XY of the ego origin
  lidar2global [[...×4]×4]              local-ego → world transform
  scores       [float, ...]             raw prediction scores (all, unfiltered)
  labels       [int, ...]               class indices
  pts_local    [[[x,y], ...], ...]      polyline points in local ego frame
  filenames    [str, ...]               camera image file paths

Usage:
    python tools/pandaset/infer.py \
        projects/configs/maptr/maptr_tiny_r50_pandaset.py \
        ckpts/maptr_tiny_r50_110e.pth \
        [--pred-dir work_dirs/maptr_tiny_r50_pandaset/preds]
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
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--pred-dir', default=None,
                   help='output directory for .npz files '
                        '(default: work_dirs/<cfg>/preds/)')
    return p.parse_args()


def main():
    args     = parse_args()
    cfg_name = osp.splitext(osp.basename(args.config))[0]
    if args.pred_dir is None:
        args.pred_dir = osp.join('work_dirs', cfg_name, 'preds')
    mmcv.mkdir_or_exist(args.pred_dir)

    logger = get_root_logger()
    logger.info(f'Predictions → {args.pred_dir}')

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin') and cfg.get('plugin_dir'):
        import importlib
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))

    cfg.model.pretrained = None
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
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pc_range = list(cfg.point_cloud_range)   # saved once per frame for draw_frames
    prog_bar = mmcv.ProgressBar(len(dataset))

    for frame_idx, data in enumerate(data_loader):
        img_metas    = data['img_metas'][0].data[0]
        lidar2global = np.array(img_metas['lidar2global'], dtype=np.float64)

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        res       = result[0]['pts_bbox']
        scores    = res['scores_3d'].numpy()
        labels    = res['labels_3d'].numpy().astype(np.int32)
        pts_local = res['pts_3d'].numpy()                         # (N, P, 2)
        filenames = img_metas.get('filename', [])

        record = {
            'ego_xy':       lidar2global[:2, 3].tolist(),
            'lidar2global': lidar2global.tolist(),
            'scores':       scores.tolist(),
            'labels':       labels.tolist(),
            'pts_local':    pts_local.tolist(),
            'filenames':    list(filenames),
            'pc_range':     pc_range,
        }
        with open(osp.join(args.pred_dir, f'frame_{frame_idx:04d}.json'), 'w') as f:
            json.dump(record, f)
        prog_bar.update()

    logger.info(f'\nDone — {len(dataset)} frames saved to {args.pred_dir}')


if __name__ == '__main__':
    main()
