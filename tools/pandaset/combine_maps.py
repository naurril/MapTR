"""
Stage 3: Combine per-frame predictions into a single global HD map.

Reads frame_NNNN.json files produced by infer.py and renders two PNGs:

  global_map.png        – result of the chosen combination method
  global_map_naive.png  – all detections above score thresh (no combination)

Available combination methods (--method):

  greedy_nms      Sort by score descending; suppress any prediction whose
                  Chamfer distance to an already-kept prediction is < --nms-dist.
                  Fast, simple, keeps the highest-confidence representative.

  dbscan          Cluster all predictions with DBSCAN (eps = --nms-dist,
                  Chamfer metric).  Keep the highest-score prediction per cluster.
                  Not order-dependent like greedy NMS; better at grouping
                  spatially close multi-frame predictions.

  weighted_merge  Same DBSCAN clustering, but instead of picking one prediction,
                  compute the score-weighted average of all polylines in each
                  cluster (with orientation alignment).  Produces geometrically
                  smoother / more stable polylines.

  growing_merge   DBSCAN clustering + PCA-based road axis estimation.  Collects
                  ALL points from every cluster member, finds the UNION of their
                  along-road extents (so the result can be longer than any single
                  prediction), then re-samples P points using a Gaussian kernel
                  for the cross-track position.  Best for long road elements
                  observed across many frames.

Usage:
    python tools/pandaset/combine_maps.py \
        [--pred-dir work_dirs/maptr_tiny_r50_pandaset/preds] \
        [--score-thresh 0.3] [--nms-dist 3.0] \
        [--method greedy_nms|dbscan|weighted_merge|growing_merge] \
        [--out work_dirs/maptr_tiny_r50_pandaset/global_map.png]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import logging
import os.path as osp

import numpy as np

from utils import (to_world, greedy_nms, dbscan_best, dbscan_weighted_merge,
                   growing_merge, render_global_map, CLASS_NAMES)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

METHODS = {
    'greedy_nms':     greedy_nms,
    'dbscan':         dbscan_best,
    'weighted_merge': dbscan_weighted_merge,
    'growing_merge':  growing_merge,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', default=None,
                   help='directory with frame_NNNN.json files '
                        '(default: work_dirs/maptr_tiny_r50_pandaset/preds)')
    p.add_argument('--score-thresh', default=0.3, type=float,
                   help='confidence threshold (applied before combination)')
    p.add_argument('--nms-dist', default=3.0, type=float,
                   help='Chamfer-distance threshold for NMS / DBSCAN eps (metres)')
    p.add_argument('--method', default='greedy_nms', choices=sorted(METHODS),
                   help='combination method (default: greedy_nms)')
    p.add_argument('--n-pts', default=None, type=int,
                   help='output points per polyline for growing_merge '
                        '(default: auto — preserves original inter-point spacing)')
    p.add_argument('--out', default=None,
                   help='output PNG path '
                        '(default: work_dirs/maptr_tiny_r50_pandaset/global_map.png)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.pred_dir is None:
        args.pred_dir = osp.join('work_dirs', 'maptr_tiny_r50_pandaset', 'preds')
    if args.out is None:
        args.out = osp.join(osp.dirname(osp.abspath(args.pred_dir)), 'global_map.png')
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)

    import functools
    combine_fn = METHODS[args.method]
    if args.method == 'growing_merge':
        combine_fn = functools.partial(combine_fn, n_out=args.n_pts)
    logger.info(f'Method: {args.method}')
    logger.info(f'Reading from: {args.pred_dir}')

    json_files = sorted(glob.glob(osp.join(args.pred_dir, 'frame_*.json')))
    if not json_files:
        raise FileNotFoundError(f'No frame_*.json files found in {args.pred_dir}')
    logger.info(f'{len(json_files)} frames')

    trajectory = []
    buckets    = [[] for _ in CLASS_NAMES]

    for json_path in json_files:
        with open(json_path) as f:
            d = json.load(f)
        trajectory.append(d['ego_xy'])

        lidar2global = np.array(d['lidar2global'])
        scores       = np.array(d['scores'])
        labels       = np.array(d['labels'], dtype=np.int32)
        pts_local    = np.array(d['pts_local'])   # (N, P, 2)

        keep = scores > args.score_thresh
        for score, label, pts in zip(scores[keep], labels[keep], pts_local[keep]):
            if int(label) < len(CLASS_NAMES):
                buckets[int(label)].append((float(score), to_world(pts, lidar2global)))

    traj = np.array(trajectory)

    logger.info(f'Combining ({args.method}) …')
    final = []
    for cls_idx, candidates in enumerate(buckets):
        kept = combine_fn(candidates, args.nms_dist)
        logger.info(f'  {CLASS_NAMES[cls_idx]:15s}: {len(candidates):5d} raw → '
                    f'{len(kept):4d} kept')
        for score, pts in kept:
            final.append((cls_idx, score, pts))

    naive = [(cls_idx, score, pts)
             for cls_idx, candidates in enumerate(buckets)
             for score, pts in candidates]

    logger.info('Rendering …')
    render_global_map(
        final, traj,
        f'Global HD Map ({args.method}) — PandaSet\n'
        f'score>{args.score_thresh}, dist={args.nms_dist}m, {len(final)} elements',
        args.out,
    )
    logger.info(f'Saved → {args.out}')

    naive_out = osp.splitext(args.out)[0] + '_naive.png'
    render_global_map(
        naive, traj,
        f'Global HD Map (naive) — PandaSet\n'
        f'score>{args.score_thresh}, {len(naive)} raw elements',
        naive_out,
    )
    logger.info(f'Saved → {naive_out}')


if __name__ == '__main__':
    main()
