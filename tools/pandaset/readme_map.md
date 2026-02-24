# MapTR HD Map Inference & Visualisation

Three-stage pipeline: **infer → draw frames → combine map**.

Scripts live in `tools/pandaset/` and are dataset-agnostic.
Run all commands from the **repo root**.

---

## PandaSet

### 1. Prepare per-sequence annotation pkl

```bash
python tools/pandaset/create_pandaset_infos.py \
    --data-root data/pandaset \
    --sequences 001 \
    --out data/pandaset/pandaset_map_infos_001.pkl

python tools/pandaset/create_pandaset_infos.py \
    --data-root data/pandaset \
    --sequences 008 \
    --out data/pandaset/pandaset_map_infos_008.pkl

python tools/pandaset/create_pandaset_infos.py \
    --data-root data/pandaset \
    --sequences 040 \
    --out data/pandaset/pandaset_map_infos_040.pkl
```

### 2. Run inference

```bash
PYTHONPATH=.:$PYTHONPATH python tools/pandaset/infer.py \
    projects/configs/maptr/maptr_tiny_r50_pandaset.py \
    ckpts/maptr_tiny_r50_110e.pth \
    --ann-file data/pandaset/pandaset_map_infos_001.pkl \
    --pred-dir work_dirs/pandaset_001/preds

PYTHONPATH=.:$PYTHONPATH python tools/pandaset/infer.py \
    projects/configs/maptr/maptr_tiny_r50_pandaset.py \
    ckpts/maptr_tiny_r50_110e.pth \
    --ann-file data/pandaset/pandaset_map_infos_008.pkl \
    --pred-dir work_dirs/pandaset_008/preds

PYTHONPATH=.:$PYTHONPATH python tools/pandaset/infer.py \
    projects/configs/maptr/maptr_tiny_r50_pandaset.py \
    ckpts/maptr_tiny_r50_110e.pth \
    --ann-file data/pandaset/pandaset_map_infos_040.pkl \
    --pred-dir work_dirs/pandaset_040/preds
```

### 3. Draw per-frame images (surround-view + BEV)

```bash
python tools/pandaset/draw_frames.py \
    --pred-dir work_dirs/pandaset_001/preds \
    --out-dir  work_dirs/pandaset_001/vis

python tools/pandaset/draw_frames.py \
    --pred-dir work_dirs/pandaset_008/preds \
    --out-dir  work_dirs/pandaset_008/vis

python tools/pandaset/draw_frames.py \
    --pred-dir work_dirs/pandaset_040/preds \
    --out-dir  work_dirs/pandaset_040/vis
```

### 4. Combine into global map

```bash
python tools/pandaset/combine_maps.py \
    --pred-dir     work_dirs/pandaset_001/preds \
    --out          work_dirs/pandaset_001/global_map.png \
    --cluster-dist 1.5

python tools/pandaset/combine_maps.py \
    --pred-dir     work_dirs/pandaset_008/preds \
    --out          work_dirs/pandaset_008/global_map.png \
    --cluster-dist 1.5

python tools/pandaset/combine_maps.py \
    --pred-dir     work_dirs/pandaset_040/preds \
    --out          work_dirs/pandaset_040/global_map.png \
    --cluster-dist 1.5
```

Outputs per sequence:
- `global_map.png` — combined map (growing_merge)
- `global_map_naive.png` — all raw detections above score threshold
- `global_map.json` — combined map data (elements + trajectory)

### 5. LiDAR + map overlay

```bash
python tools/pandaset/draw_lidar_map.py --sequence 001
python tools/pandaset/draw_lidar_map.py --sequence 008
python tools/pandaset/draw_lidar_map.py --sequence 040
```

Output: `work_dirs/pandaset_<seq>/lidar_map.png`

---

## nuScenes-mini

The val pkl (`data/nuscenes/nuscenes_infos_temporal_val.pkl`) contains
2 scenes (40 + 41 frames).

### 1. Run inference

```bash
PYTHONPATH=.:$PYTHONPATH python tools/pandaset/infer.py \
    projects/configs/maptr/maptr_tiny_r50_24e.py \
    ckpts/maptr_tiny_r50_110e.pth \
    --pred-dir work_dirs/nuscenes_mini/preds
```

### 2. Combine + draw (per scene)

Combines each scene separately, renders both plain map and LiDAR overlay:

```bash
python tools/pandaset/draw_lidar_map_nusc.py \
    --pred-dir  work_dirs/nuscenes_mini/preds \
    --info-pkl  data/nuscenes/nuscenes_infos_temporal_val.pkl \
    --out-dir   work_dirs/nuscenes_mini \
    --cluster-dist 1.0
```

Outputs per scene (`<short>` = first 8 chars of scene token):
- `scene_<short>_map.png` — combined map
- `scene_<short>_map_naive.png` — naive (no combination)
- `scene_<short>_lidar_map.png` — LiDAR point cloud + map overlay
- `scene_<short>_map.json` — combined map data
- `scene_<short>_map_naive.json` — naive map data

---

## Key parameters

| Flag | Default | Description |
|---|---|---|
| `--score-thresh` | `0.3` | Confidence threshold for predictions |
| `--nms-dist` | `3.0` | Chamfer distance for greedy NMS / DBSCAN (non-growing methods) |
| `--cluster-dist` | `nms-dist/2` | DBSCAN eps for `growing_merge`; keep tighter than `nms-dist` to avoid merging parallel elements |
| `--method` | `growing_merge` | Combination method: `growing_merge`, `weighted_merge`, `dbscan`, `greedy_nms` |
| `--n-pts` | auto | Output points per polyline for `growing_merge` (auto = preserves input spacing) |
| `--cols` | `3` | Camera columns in surround-view mosaic |
