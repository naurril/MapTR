"""
Stage 2: Render per-frame surround-view and BEV map images.

Reads frame_NNNN.json files produced by infer.py and writes two images per
frame into a single flat output directory (no sub-folders):

  frame_NNNN_surround.jpg   – surround-view camera mosaic
  frame_NNNN_map.png        – BEV prediction plot (local ego frame)

Dataset-agnostic: camera images are laid out in the order stored by infer.py
(which matches the config's camera list), with --cols images per row.

Usage:
    python tools/pandaset/draw_frames.py \
        [--pred-dir work_dirs/<run>/preds] \
        [--score-thresh 0.3] [--cols 3] \
        [--out-dir work_dirs/<run>/vis]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import os.path as osp

import logging

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import make_surround_view, plot_bev

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

CAR_ICON = 'figs/lidar_car.png'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', default=None,
                   help='directory with frame_NNNN.npz files '
                        '(default: work_dirs/maptr_tiny_r50_pandaset/preds)')
    p.add_argument('--score-thresh', default=0.3, type=float)
    p.add_argument('--cols', default=3, type=int,
                   help='camera images per row in the surround-view mosaic (default 3)')
    p.add_argument('--out-dir', default=None,
                   help='output directory for images '
                        '(default: <pred-dir>/../vis)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.pred_dir is None:
        args.pred_dir = osp.join('work_dirs', 'maptr_tiny_r50_pandaset', 'preds')
    if args.out_dir is None:
        args.out_dir = osp.join(osp.dirname(osp.abspath(args.pred_dir)), 'vis')
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f'Reading from: {args.pred_dir}')
    logger.info(f'Writing to:   {args.out_dir}')

    npz_files = sorted(glob.glob(osp.join(args.pred_dir, 'frame_*.json')))
    if not npz_files:
        raise FileNotFoundError(f'No frame_*.json files found in {args.pred_dir}')
    logger.info(f'{len(npz_files)} frames')

    car_img = Image.open(CAR_ICON) if osp.isfile(CAR_ICON) else None

    for npz_path in tqdm(npz_files, desc='frames'):
        frame_id = osp.splitext(osp.basename(npz_path))[0]   # e.g. frame_0042
        with open(npz_path) as f:
            d = json.load(f)

        # ── surround-view mosaic ───────────────────────────────────────────
        mosaic = make_surround_view(d['filenames'], cols=args.cols)
        cv2.imwrite(osp.join(args.out_dir, f'{frame_id}_surround.jpg'),
                    mosaic, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # ── BEV prediction map ─────────────────────────────────────────────
        scores    = np.array(d['scores'])
        labels    = np.array(d['labels'], dtype=np.int32)
        pts_local = np.array(d['pts_local'])   # (N, P, 2)
        keep      = scores > args.score_thresh
        plot_bev(pts_local[keep], labels[keep], scores[keep],
                 osp.join(args.out_dir, f'{frame_id}_map.png'),
                 pc_range=d.get('pc_range'),
                 car_img=car_img)

    logger.info(f'Done — images saved to {args.out_dir}')


if __name__ == '__main__':
    main()
