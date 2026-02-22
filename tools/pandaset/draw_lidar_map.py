"""
Overlay the global HD map on a top-down accumulated point-cloud for one
PandaSet sequence.

Reads:
  - data/pandaset/<seq>/lidar/<frame>.pkl  (world-frame x,y,z)
  - <map-json>  produced by combine_maps.py  (global_map.json)

Writes:
  - <out>  a single PNG with the point cloud as background and map on top

Usage:
    python tools/pandaset/draw_lidar_map.py \
        --sequence 001 \
        [--data-root data/pandaset] \
        [--map-json work_dirs/pandaset_001/global_map.json] \
        [--out     work_dirs/pandaset_001/lidar_map.png] \
        [--max-pts 2000000]
"""
import argparse
import glob
import json
import os
import os.path as osp
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

CLASS_COLORS = ['#FF8C00', '#1E90FF', '#32CD32']
CLASS_NAMES  = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sequence',  required=True,
                   help='PandaSet sequence ID, e.g. 001')
    p.add_argument('--data-root', default='data/pandaset')
    p.add_argument('--map-json',  default=None,
                   help='global_map.json from combine_maps.py '
                        '(default: work_dirs/pandaset_<seq>/global_map.json)')
    p.add_argument('--out',       default=None,
                   help='output PNG path '
                        '(default: work_dirs/pandaset_<seq>/lidar_map.png)')
    p.add_argument('--max-pts',   default=2_000_000, type=int,
                   help='subsample if total points exceed this (default 2M)')
    return p.parse_args()


def main():
    args = parse_args()
    seq = args.sequence
    run_dir = osp.join('work_dirs', f'pandaset_{seq}')

    if args.map_json is None:
        args.map_json = osp.join(run_dir, 'global_map.json')
    if args.out is None:
        args.out = osp.join(run_dir, 'lidar_map.png')
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)

    # ── load global map ────────────────────────────────────────────────────
    with open(args.map_json) as f:
        map_data = json.load(f)
    elements   = map_data['elements']
    trajectory = np.array(map_data['trajectory'])

    # ── accumulate lidar point clouds ──────────────────────────────────────
    lidar_dir = osp.join(args.data_root, seq, 'lidar')
    pkl_files = sorted(glob.glob(osp.join(lidar_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f'No lidar pkl files found in {lidar_dir}')

    all_xy = []
    for pkl_path in tqdm(pkl_files, desc='Loading lidar'):
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
        all_xy.append(df[['x', 'y']].values.astype(np.float32))

    all_xy = np.concatenate(all_xy, axis=0)

    # Subsample if needed
    if len(all_xy) > args.max_pts:
        idx    = np.random.choice(len(all_xy), args.max_pts, replace=False)
        all_xy = all_xy[idx]

    # ── render ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')
    ax.set_facecolor('#0d0d0d')
    fig.patch.set_facecolor('#0d0d0d')

    # Point cloud as 2-D histogram (fast, avoids millions of scatter points)
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    span       = max(xmax - xmin, ymax - ymin)
    bins       = int(np.clip(span / 0.2, 200, 2000))   # ~0.2 m/bin
    h, xe, ye  = np.histogram2d(all_xy[:, 0], all_xy[:, 1],
                                 bins=bins,
                                 range=[[xmin, xmax], [ymin, ymax]])
    # Log-scale so sparse regions are still visible
    h = np.log1p(h).T
    ax.imshow(h, origin='lower', cmap='Blues',
              extent=[xmin, xmax, ymin, ymax],
              aspect='equal', interpolation='nearest', zorder=1)

    # Ego trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            color='white', linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(trajectory[0,  0], trajectory[0,  1],
               color='white', s=40, zorder=3, marker='o')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
               color='yellow', s=40, zorder=3, marker='*')

    # Map elements
    for elem in elements:
        pts   = np.array(elem['pts'])
        color = CLASS_COLORS[elem['class_idx']]
        ax.plot(pts[:, 0], pts[:, 1],
                color=color, linewidth=1.5,
                alpha=min(1.0, elem['score'] + 0.2), zorder=4)

    # Legend
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
    ax.set_title(f'PandaSet {seq} — LiDAR + HD Map\n'
                 f'{len(pkl_files)} frames, {len(elements)} map elements',
                 color='white', fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
