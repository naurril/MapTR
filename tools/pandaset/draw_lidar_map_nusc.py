"""
Overlay the combined HD map on a top-down accumulated LiDAR point cloud
for each nuScenes scene in a pred-dir.

For each scene found in the frame JSONs:
  1. Combines per-frame predictions with growing_merge.
  2. Loads the LIDAR_TOP .pcd.bin files (local lidar frame) and transforms
     them to world frame using the lidar2global matrix from the frame JSON.
  3. Renders LiDAR point cloud + map + ego trajectory into one PNG.

Lidar paths are looked up from the nuScenes info pkl (the frame JSONs carry
scene_token for grouping but not lidar_path).

Usage:
    python tools/pandaset/draw_lidar_map_nusc.py \
        [--pred-dir  work_dirs/nuscenes_mini/preds] \
        [--info-pkl  data/nuscenes/nuscenes_infos_temporal_val.pkl] \
        [--out-dir   work_dirs/nuscenes_mini] \
        [--score-thresh 0.3] [--nms-dist 3.0] [--max-pts 2000000]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from utils import (to_world, growing_merge, CLASS_NAMES, CLASS_COLORS,
                   render_global_map)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir',  default='work_dirs/nuscenes_mini/preds')
    p.add_argument('--info-pkl',  default='data/nuscenes/nuscenes_infos_temporal_val.pkl')
    p.add_argument('--out-dir',   default=None,
                   help='output directory (default: pred-dir/..)')
    p.add_argument('--score-thresh',  default=0.3,  type=float)
    p.add_argument('--nms-dist',      default=3.0,  type=float)
    p.add_argument('--cluster-dist',  default=None, type=float,
                   help='DBSCAN eps for growing_merge (default: nms-dist/2)')
    p.add_argument('--max-pts',       default=2_000_000, type=int)
    return p.parse_args()


def load_lidar_world(lidar_path, lidar2global):
    """Load a nuScenes .pcd.bin, transform xyz to world frame, return (N,2)."""
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    n   = len(pts)
    h   = np.ones((n, 4), dtype=np.float64)
    h[:, :3] = pts.astype(np.float64)
    return (lidar2global @ h.T).T[:, :2].astype(np.float32)


def combine_scene(frame_jsons, score_thresh, nms_dist, cluster_dist=None):
    """Run growing_merge over all frames in one scene.
    Returns (elements, naive_elements, trajectory).
    """
    trajectory = []
    buckets    = [[] for _ in CLASS_NAMES]

    for jp in frame_jsons:
        with open(jp) as f:
            d = json.load(f)
        trajectory.append(d['ego_xy'])
        lidar2global = np.array(d['lidar2global'])
        scores       = np.array(d['scores'])
        labels       = np.array(d['labels'], dtype=np.int32)
        pts_local    = np.array(d['pts_local'])

        keep = scores > score_thresh
        for score, label, pts in zip(scores[keep], labels[keep], pts_local[keep]):
            if int(label) < len(CLASS_NAMES):
                buckets[int(label)].append(
                    (float(score), to_world(pts, lidar2global)))

    elements = []
    for cls_idx, candidates in enumerate(buckets):
        for score, pts in growing_merge(candidates, nms_dist, cluster_dist=cluster_dist):
            elements.append({'class': CLASS_NAMES[cls_idx],
                             'class_idx': cls_idx,
                             'score': float(score),
                             'pts': pts})

    naive = [{'class': CLASS_NAMES[cls_idx], 'class_idx': cls_idx,
              'score': float(score), 'pts': pts}
             for cls_idx, candidates in enumerate(buckets)
             for score, pts in candidates]

    return elements, naive, np.array(trajectory)


def render(all_xy, elements, trajectory, title, out_path, max_pts):
    if len(all_xy) > max_pts:
        idx    = np.random.choice(len(all_xy), max_pts, replace=False)
        all_xy = all_xy[idx]

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')
    ax.set_facecolor('#0d0d0d')
    fig.patch.set_facecolor('#0d0d0d')

    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    span  = max(xmax - xmin, ymax - ymin)
    bins  = int(np.clip(span / 0.2, 200, 2000))
    h, xe, ye = np.histogram2d(all_xy[:, 0], all_xy[:, 1], bins=bins,
                                range=[[xmin, xmax], [ymin, ymax]])
    ax.imshow(np.log1p(h).T, origin='lower', cmap='Blues',
              extent=[xmin, xmax, ymin, ymax],
              aspect='equal', interpolation='nearest', zorder=1)

    ax.plot(trajectory[:, 0], trajectory[:, 1],
            color='white', linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(trajectory[0,  0], trajectory[0,  1],
               color='white',  s=40, zorder=3, marker='o')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
               color='yellow', s=40, zorder=3, marker='*')

    for elem in elements:
        pts   = np.array(elem['pts'])
        color = CLASS_COLORS[elem['class_idx']]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                alpha=min(1.0, elem['score'] + 0.2), zorder=4)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    patches.append(mpatches.Patch(color='white', label='ego trajectory'))
    ax.legend(handles=patches, loc='upper right',
              facecolor='#1a1a2e', edgecolor='white',
              labelcolor='white', fontsize=9)
    ax.set_xlabel('World X (m)', color='white')
    ax.set_ylabel('World Y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.set_title(title, color='white', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved → {out_path}')


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = osp.dirname(osp.abspath(args.pred_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load info pkl to get lidar paths ──────────────────────────────────
    with open(args.info_pkl, 'rb') as f:
        infos = pickle.load(f)['infos']
    # index infos by scene_token, preserving frame order
    info_by_scene = defaultdict(list)
    for info in infos:
        info_by_scene[info['scene_token']].append(info)

    # ── group frame JSONs by scene_token ───────────────────────────────────
    frame_jsons = sorted(glob.glob(osp.join(args.pred_dir, 'frame_*.json')))
    if not frame_jsons:
        raise FileNotFoundError(f'No frame_*.json in {args.pred_dir}')

    frames_by_scene = defaultdict(list)
    for jp in frame_jsons:
        with open(jp) as f:
            st = json.load(f).get('scene_token', '')
        frames_by_scene[st].append(jp)

    # ── process each scene ─────────────────────────────────────────────────
    for scene_token, scene_frames in frames_by_scene.items():
        short = scene_token[:8]
        print(f'\nScene {short}  ({len(scene_frames)} frames)')

        scene_infos = info_by_scene.get(scene_token, [])
        if len(scene_infos) != len(scene_frames):
            print(f'  WARNING: info count {len(scene_infos)} ≠ frame count '
                  f'{len(scene_frames)} — skipping lidar load')
            scene_infos = []

        # combine map
        elements, naive, trajectory = combine_scene(
            scene_frames, args.score_thresh, args.nms_dist, args.cluster_dist)
        print(f'  Map elements: {len(elements)} combined, {len(naive)} naive')

        # accumulate lidar
        all_xy_parts = []
        for frame_json, info in tqdm(
                zip(scene_frames, scene_infos), total=len(scene_frames),
                desc='  Loading lidar'):
            lidar_path = info.get('lidar_path', '')
            if not lidar_path or not osp.isfile(lidar_path):
                continue
            with open(frame_json) as f:
                d = json.load(f)
            lidar2global = np.array(d['lidar2global'])
            all_xy_parts.append(load_lidar_world(lidar_path, lidar2global))

        if not all_xy_parts:
            print('  No lidar data found, skipping.')
            continue
        all_xy = np.concatenate(all_xy_parts, axis=0)

        # plain map PNG
        map_png = osp.join(args.out_dir, f'scene_{short}_map.png')
        render_global_map(
            [(e['class_idx'], e['score'], np.array(e['pts'])) for e in elements],
            trajectory,
            f'nuScenes {short} — HD Map (growing_merge)\n'
            f'{len(scene_frames)} frames, {len(elements)} elements',
            map_png,
        )
        print(f'Saved → {map_png}')

        # naive map PNG
        naive_png = osp.join(args.out_dir, f'scene_{short}_map_naive.png')
        render_global_map(
            [(e['class_idx'], e['score'], np.array(e['pts'])) for e in naive],
            trajectory,
            f'nuScenes {short} — HD Map (naive)\n'
            f'{len(scene_frames)} frames, {len(naive)} raw elements',
            naive_png,
        )
        print(f'Saved → {naive_png}')

        # lidar + map overlay PNG
        title = (f'nuScenes {short} — LiDAR + HD Map\n'
                 f'{len(scene_frames)} frames, {len(elements)} map elements')
        out_path = osp.join(args.out_dir, f'scene_{short}_lidar_map.png')
        render(all_xy, elements, trajectory, title, out_path, args.max_pts)

        # save combined map JSON
        map_json_path = osp.join(args.out_dir, f'scene_{short}_map.json')
        with open(map_json_path, 'w') as f:
            json.dump({
                'scene_token': scene_token,
                'method': 'growing_merge',
                'score_thresh': args.score_thresh,
                'nms_dist': args.nms_dist,
                'trajectory': trajectory.tolist(),
                'elements': [
                    {**e, 'pts': np.array(e['pts']).tolist()}
                    for e in elements
                ],
            }, f, indent=2)
        print(f'Saved → {map_json_path}')

        # save naive map JSON
        naive_json_path = osp.join(args.out_dir, f'scene_{short}_map_naive.json')
        with open(naive_json_path, 'w') as f:
            json.dump({
                'scene_token': scene_token,
                'method': 'naive',
                'score_thresh': args.score_thresh,
                'trajectory': trajectory.tolist(),
                'elements': [
                    {**e, 'pts': np.array(e['pts']).tolist()}
                    for e in naive
                ],
            }, f, indent=2)
        print(f'Saved → {naive_json_path}')


if __name__ == '__main__':
    main()
