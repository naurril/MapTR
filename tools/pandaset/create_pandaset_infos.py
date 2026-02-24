"""
Convert PandaSet sequence(s) into a MapTR-compatible info pkl.

Usage:
    python tools/pandaset/create_pandaset_infos.py \
        --data-root data/pandaset \
        --sequences 008 \
        --out-dir data/pandaset
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

# Map PandaSet camera folder names to nuScenes-style CAM keys.
# Order MUST match nuScenes camera order so the model's learned cams_embeds
# (one per slot index) are applied to the geometrically correct camera.
# nuScenes order: FRONT(0), FRONT_RIGHT(1), FRONT_LEFT(2), BACK(3),
#                 BACK_LEFT(4), BACK_RIGHT(5)
PANDASET_CAM_MAP = {
    'front_camera':       'CAM_FRONT',
    'front_right_camera': 'CAM_FRONT_RIGHT',  # slot 1 — matches nuScenes slot 1
    'front_left_camera':  'CAM_FRONT_LEFT',   # slot 2 — matches nuScenes slot 2
    'back_camera':        'CAM_BACK',
    'left_camera':        'CAM_BACK_LEFT',
    'right_camera':       'CAM_BACK_RIGHT',
}


def quat_dict_to_rot(q):
    """Convert {w, x, y, z} dict → 3x3 rotation matrix (float64)."""
    # scipy uses scalar-last convention [x, y, z, w]
    return Rscipy.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()


def rot_to_quat_wxyz(mat):
    """Convert 3x3 rotation matrix → [w, x, y, z] list."""
    q = Rscipy.from_matrix(mat).as_quat()  # [x, y, z, w]
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def build_sequence_infos(seq_path: Path, seq_id: str):
    seq_path = Path(seq_path)

    with open(seq_path / 'meta' / 'timestamps.json') as f:
        timestamps = json.load(f)
    with open(seq_path / 'meta' / 'gps.json') as f:
        gps = json.load(f)
    with open(seq_path / 'lidar' / 'poses.json') as f:
        lidar_poses = json.load(f)

    # Pre-load per-camera intrinsics and per-frame poses
    cam_intrinsics = {}
    cam_poses = {}
    for panda_name in PANDASET_CAM_MAP:
        cam_dir = seq_path / 'camera' / panda_name
        with open(cam_dir / 'intrinsics.json') as f:
            d = json.load(f)
        cam_intrinsics[panda_name] = np.array(
            [[d['fx'], 0,       d['cx']],
             [0,       d['fy'], d['cy']],
             [0,       0,       1      ]], dtype=np.float32)
        with open(cam_dir / 'poses.json') as f:
            cam_poses[panda_name] = json.load(f)

    n_frames = len(timestamps)
    infos = []

    for i in range(n_frames):
        lp = lidar_poses[i]
        R_L = quat_dict_to_rot(lp['heading'])
        t_L = np.array([lp['position']['x'],
                         lp['position']['y'],
                         lp['position']['z']])

        ego2global_rotation = [lp['heading']['w'], lp['heading']['x'],
                                lp['heading']['y'], lp['heading']['z']]
        ego2global_translation = t_L.tolist()

        # 18-element can_bus (fields 0-6 are overwritten by get_data_info)
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[:3] = t_L
        can_bus[3:7] = ego2global_rotation
        if i < len(gps):
            can_bus[7] = gps[i].get('xvel', 0.0)
            can_bus[8] = gps[i].get('yvel', 0.0)

        cams = {}
        for panda_name, maptr_name in PANDASET_CAM_MAP.items():
            cp = cam_poses[panda_name][i]
            R_C = quat_dict_to_rot(cp['heading'])
            t_C = np.array([cp['position']['x'],
                             cp['position']['y'],
                             cp['position']['z']])

            # MapTR's BEV encoder generates reference points in the LOCAL lidar
            # body frame and projects them via lidar2img → camera.
            # sensor2lidar must therefore be camera-to-lidar-body:
            #   p_cam = R_C.T @ R_L @ p_local + R_C.T @ (t_L - t_C)
            #         = R_C.T @ (R_L @ p_local + t_L - t_C)
            # This is achieved by:
            #   sensor2lidar_rotation    = R_L.T @ R_C
            #   sensor2lidar_translation = R_L.T @ (t_C - t_L)
            # (poses.json is sensor-to-world for both lidar and camera)
            s2l_rot = (R_L.T @ R_C).astype(np.float32)
            s2l_t   = (R_L.T @ (t_C - t_L)).astype(np.float32)

            cams[maptr_name] = {
                'data_path': str(seq_path / 'camera' / panda_name / f'{i:02d}.jpg'),
                'sensor2lidar_rotation':    s2l_rot,
                'sensor2lidar_translation': s2l_t,
                'cam_intrinsic':            cam_intrinsics[panda_name],
                'sensor2ego_rotation':    rot_to_quat_wxyz(s2l_rot),
                'sensor2ego_translation': s2l_t.tolist(),
            }

        token = f'pandaset_{seq_id}_{i:03d}'
        infos.append({
            'token':      token,
            'scene_token': f'pandaset_{seq_id}',
            'timestamp':  timestamps[i],
            'frame_idx':  i,
            'prev': f'pandaset_{seq_id}_{i-1:03d}' if i > 0 else '',
            'next': f'pandaset_{seq_id}_{i+1:03d}' if i < n_frames - 1 else '',
            'lidar_path': str(seq_path / 'lidar' / f'{i:02d}.pkl'),
            'sweeps': [],
            'cams': cams,
            'ego2global_translation': ego2global_translation,
            'ego2global_rotation':    ego2global_rotation,
            # LiDAR IS the ego frame → identity
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation':    [1.0, 0.0, 0.0, 0.0],
            'can_bus':      can_bus,
            'map_location': 'boston-seaport',  # dummy; not used in test-only mode
        })

    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/pandaset')
    parser.add_argument('--sequences', nargs='+', default=['008'])
    parser.add_argument('--out-dir', default='data/pandaset')
    parser.add_argument('--out', default=None,
                        help='explicit output pkl path (overrides --out-dir default)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_infos = []
    for seq_id in args.sequences:
        print(f'Processing sequence {seq_id} …')
        infos = build_sequence_infos(data_root / seq_id, seq_id)
        all_infos.extend(infos)
        print(f'  → {len(infos)} frames')

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.out_dir) / 'pandaset_map_infos_test.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'infos': all_infos, 'metadata': {'version': 'pandaset'}}, f)
    print(f'Saved {len(all_infos)} frames to {out_path}')


if __name__ == '__main__':
    main()
