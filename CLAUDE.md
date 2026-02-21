# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MapTR is an end-to-end framework for vectorized HD map construction from multi-view camera images for autonomous driving. It models map elements (lane dividers, pedestrian crossings, road boundaries) as permutation-equivalent point sets and decodes them using a hierarchical transformer. The `maptrv2` branch extends the original with centerline support, Argoverse2 dataset, and one-to-many matching.

## Build & Install

```bash
# 1. Install mmdetection3d (CUDA extensions compiled here)
cd mmdetection3d && python setup.py develop

# 2. Install Geometric Kernel Attention CUDA op
cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && python setup.py build install

# 3. Install Python dependencies
pip install -r requirement.txt
```

Key dependency versions: PyTorch>=1.9.1+cu111, mmcv-full==1.4.0, mmdet==2.14.0, mmsegmentation==0.14.1, shapely==1.8.5.post1.

## Common Commands

### Training
```bash
# Distributed training (8 GPUs)
./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8

# Single-GPU (for debugging)
PYTHONPATH=".":$PYTHONPATH python tools/train.py ./projects/configs/maptr/maptr_tiny_r50_24e.py --gpus 1
```

### Evaluation
```bash
# Distributed eval with Chamfer distance metric
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py ./path/to/ckpt.pth 8
```

### Dataset Preparation
```bash
# nuScenes
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data

# Argoverse2
python tools/maptrv2/custom_av2_map_converter.py --data-root ./data/argoverse2/sensor/
```

### Visualization
```bash
python tools/maptrv2/nusc_vis_pred.py <config> <checkpoint> --score-thresh 0.3
python tools/maptrv2/av2_vis_pred.py <config> <checkpoint> --score-thresh 0.3
python tools/maptr/generate_video.py /path/to/vis_pred/
```

## Architecture

### Framework Stack
Built as an **mmdetection3d plugin**. The plugin (`projects/mmdet3d_plugin/`) registers custom components into mmcv's registry system (DETECTORS, HEADS, LOSSES, etc.). All model configs are Python files in `projects/configs/`. The `PYTHONPATH` must include the repo root so the plugin is discovered.

### Core Pipeline
```
Multi-view images (6 cameras) → ResNet backbone → FPN neck → BEV feature extraction → Transformer decoder → Vectorized map output
```

### Key Source Locations

- **Detectors**: `projects/mmdet3d_plugin/maptr/detectors/` — `MapTR` and `MapTRv2` classes (extend `MVXTwoStageDetector`)
- **Detection heads**: `projects/mmdet3d_plugin/maptr/dense_heads/` — `MapTRHead` and `MapTRv2Head` produce class logits + point coordinates
- **Transformer**: `projects/mmdet3d_plugin/maptr/modules/transformer.py` — `MapTRPerceptionTransformer` combining BEV encoder and map decoder
- **Decoder**: `projects/mmdet3d_plugin/maptr/modules/decoder.py` — `MapTRDecoder` with iterative refinement across 6 layers
- **BEV encoder**: `projects/mmdet3d_plugin/bevformer/` — `BEVFormerEncoder` with temporal self-attention and geometry spatial cross-attention
- **GKT attention**: `projects/mmdet3d_plugin/maptr/modules/geometry_kernel_attention.py` — custom attention with CUDA kernel in `modules/ops/geometric_kernel_attn/`
- **Losses**: `projects/mmdet3d_plugin/maptr/losses/` — `PtsL1Loss`, `PtsDirCosLoss`
- **Matching**: `projects/mmdet3d_plugin/maptr/assigners/` — Hungarian bipartite matching (`MapTRAssigner`)
- **Datasets**: `projects/mmdet3d_plugin/datasets/` — `CustomNuScenesLocalMapDataset`, `Argoverse2MapDataset`
- **Data converters**: `tools/maptrv2/` — generate temporal annotation pickle files from raw datasets

### Hierarchical Query Design
Each map element is represented by an instance-level query decomposed into point-level sub-queries (default: 50 vectors × 20 points). The decoder uses permutation-equivalent matching so point ordering within an element doesn't affect the loss.

### Config System
Configs inherit from `projects/configs/_base_/` and are organized by model variant:
- `projects/configs/maptr/` — MapTR v1 configs (nuScenes, AV2, various backbones)
- `projects/configs/maptrv2/` — MapTRv2 configs (with/without centerline)
- `projects/configs/bevformer/` — BEVFormer baseline configs

Config naming convention: `{model}_{backbone}_{dataset}_{epochs}[_variant].py`

### Multi-modal Fusion
`MapTRv2` supports camera+LiDAR fusion via `ConvFuser` in `projects/mmdet3d_plugin/maptr/modules/builder.py`. The LiDAR branch uses `LSSTransform` (Lift-Splat-Shoot) to produce BEV features that are concatenated with camera BEV features.

## Data Layout
```
data/
├── can_bus/                    # nuScenes CAN bus expansion
├── nuscenes/
│   ├── maps/, samples/, sweeps/, v1.0-trainval/
│   └── nuscenes_map_infos_temporal_{train,val}.pkl  # generated annotations
└── argoverse2/sensor/{train,val,test}/
```

Pretrained weights go in `ckpts/` (ResNet50, ResNet18, etc.).
