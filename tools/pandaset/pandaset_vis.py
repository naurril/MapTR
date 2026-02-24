"""
Run MapTR inference on PandaSet and visualise predictions.

Usage:
    python tools/pandaset/pandaset_vis.py \
        projects/configs/maptr/maptr_tiny_r50_pandaset.py \
        ckpts/maptr_tiny_r50_110e.pth \
        [--score-thresh 0.3] [--show-dir work_dirs/pandaset_vis]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import projects.mmdet3d_plugin.compat  # noqa: F401  install v1->v2 shims

import argparse
import os.path as osp

import cv2
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# PandaSet camera name → nuScenes-style key
PANDASET_CAM_MAP = {
    'front_camera':       'CAM_FRONT',
    'front_left_camera':  'CAM_FRONT_LEFT',
    'front_right_camera': 'CAM_FRONT_RIGHT',
    'back_camera':        'CAM_BACK',
    'left_camera':        'CAM_BACK_LEFT',
    'right_camera':       'CAM_BACK_RIGHT',
}
# Reverse: CAM_* → panda folder name
CAM_TO_PANDA = {v: k for k, v in PANDASET_CAM_MAP.items()}

# Surround-view camera order (top row: front; bottom row: back)
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']

# BEV map element colours: divider=orange, ped_crossing=blue, boundary=green
COLORS = ['orange', 'b', 'g']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='test config file path')
    p.add_argument('checkpoint', help='checkpoint .pth file')
    p.add_argument('--score-thresh', default=0.3, type=float,
                   help='confidence threshold for predictions')
    p.add_argument('--show-dir', default=None,
                   help='directory to save outputs (default: work_dirs/<cfg>/vis_pred)')
    return p.parse_args()


def build_cam_path_dict(filename_list):
    """Map CAM_* name → image file path from img_metas filename list."""
    cam_paths = {}
    for path in filename_list:
        # path: …/camera/<panda_cam_name>/<frame>.jpg
        panda_name = osp.basename(osp.dirname(path))
        cam_key = PANDASET_CAM_MAP.get(panda_name)
        if cam_key:
            cam_paths[cam_key] = path
    return cam_paths


def make_surround_view(cam_paths, out_path):
    """Stack 6 camera images into a 2×3 surround-view mosaic."""
    rows = []
    for row_cams in [CAMS[:3], CAMS[3:]]:
        imgs = []
        for cam in row_cams:
            img_path = cam_paths.get(cam)
            if img_path and osp.isfile(img_path):
                img = cv2.imread(img_path)
            else:
                img = np.zeros((270, 480, 3), dtype=np.uint8)
            imgs.append(img)
        # resize to common height within the row
        h = min(im.shape[0] for im in imgs)
        imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                for im in imgs]
        rows.append(cv2.hconcat(imgs))
    # make the two rows the same width
    w = min(r.shape[1] for r in rows)
    rows = [r[:, :w] for r in rows]
    mosaic = cv2.vconcat(rows)
    cv2.imwrite(out_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 70])


def plot_pred(result, pc_range, car_img, score_thresh, out_path):
    """Save a BEV plot of predicted map elements."""
    fig, ax = plt.subplots(figsize=(2, 4))
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.axis('off')

    res = result[0]['pts_bbox']
    scores  = res['scores_3d']
    labels  = res['labels_3d']
    pts_all = res['pts_3d']
    keep    = scores > score_thresh

    for score, label, pts in zip(scores[keep], labels[keep], pts_all[keep]):
        pts = pts.numpy()
        ax.plot(pts[:, 0], pts[:, 1],
                color=COLORS[label], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=COLORS[label], s=1, alpha=0.8, zorder=-1)

    if car_img is not None:
        ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    fig.savefig(out_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # load plugin
    if cfg.get('plugin') and cfg.get('plugin_dir'):
        import importlib
        parts = cfg.plugin_dir.rstrip('/').split('/')
        mod_path = '.'.join(parts[:-1]) if parts[-1] == '' else '.'.join(parts)
        # strip trailing 'mmdet3d_plugin' component to get package
        # plugin_dir = 'projects/mmdet3d_plugin/'
        _dir = cfg.plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(_dir)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # force test_mode
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        samples_per_gpu = 1

    # output directory
    if args.show_dir is None:
        args.show_dir = osp.join(
            'work_dirs',
            osp.splitext(osp.basename(args.config))[0],
            'vis_pred')
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))

    logger = get_root_logger()
    logger.info(f'Output directory: {args.show_dir}')

    # dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    nonshuffler_sampler = cfg.data.get('nonshuffler_sampler',
                                        dict(type='DistributedSampler'))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=nonshuffler_sampler,
    )
    logger.info(f'Dataset: {len(dataset)} frames')

    # model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = (checkpoint.get('meta', {}).get('CLASSES')
                     or getattr(dataset, 'CLASSES', None))
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pc_range = cfg.point_cloud_range

    # optional car icon overlay
    car_icon_path = 'figs/lidar_car.png'
    car_img = Image.open(car_icon_path) if osp.isfile(car_icon_path) else None

    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data[0]

        # derive sample directory name from pts_filename
        pts_filename = img_metas.get('pts_filename', f'frame_{i:04d}.pkl')
        # e.g. data/pandaset/008/lidar/00.pkl → seq=008, frame=00
        parts = pts_filename.replace('\\', '/').split('/')
        try:
            seq_id    = parts[-3]   # '008'
            frame_id  = osp.splitext(parts[-1])[0]  # '00'
            dir_name  = f'pandaset_{seq_id}_{frame_id}'
        except IndexError:
            dir_name  = f'frame_{i:04d}'

        sample_dir = osp.join(args.show_dir, dir_name)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        # build camera path dict
        filename_list = img_metas.get('filename', [])
        cam_paths = build_cam_path_dict(filename_list)

        # save individual camera images
        for cam_key, src_path in cam_paths.items():
            dst = osp.join(sample_dir, f'{cam_key}.jpg')
            if osp.isfile(src_path) and not osp.isfile(dst):
                import shutil
                shutil.copyfile(src_path, dst)

        # surround-view mosaic
        surround_path = osp.join(sample_dir, 'surround_view.jpg')
        make_surround_view(
            {cam: osp.join(sample_dir, f'{cam}.jpg') for cam in CAMS},
            surround_path)

        # inference
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # prediction BEV plot
        pred_path = osp.join(sample_dir, 'PRED_MAP.png')
        plot_pred(result, pc_range, car_img, args.score_thresh, pred_path)

        prog_bar.update()

    logger.info(f'\nDone. Results saved to {args.show_dir}')


if __name__ == '__main__':
    main()
