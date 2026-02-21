"""Minimal nuScenes-mini data converter, bypasses create_gt_database imports."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_converter'))

import nuscenes_converter as nuscenes_converter

root_path = './data/nuscenes'
out_dir = './data/nuscenes'
can_bus_root_path = './data'
info_prefix = 'nuscenes'
version = 'v1.0-mini'
max_sweeps = 10

print(f'Creating nuScenes infos for {version} ...')
nuscenes_converter.create_nuscenes_infos(
    root_path, out_dir, can_bus_root_path, info_prefix,
    version=version, max_sweeps=max_sweeps)

from os import path as osp
info_train_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
info_val_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')

print('Exporting 2D annotations (train) ...')
nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
print('Exporting 2D annotations (val) ...')
nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)
print('Done.')
