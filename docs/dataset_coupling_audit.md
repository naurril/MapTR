# MapTR / MapTRv2 — Dataset & Sensor Coupling Audit

This document records every place in the MapTR/MapTRv2 codebase where a design
decision is tied to a specific dataset, sensor rig, coordinate system, or
calibration geometry.  Each finding states whether it is a **config parameter**
(flexible without retraining), a **checkpoint-locked weight shape** (requires
retraining or surgery), or a **silent semantic coupling** (runs without crashing
but produces wrong results on a different dataset).

Findings were produced by static analysis of the codebase and validated against
experiments moving a nuScenes-trained model to PandaSet.

---

## Background: Two Fundamentally Different BEV-Lifting Approaches

MapTR supports three encoder backends, selectable via `encoder.type` in config:

| Backend | BEV lifting mechanism | Sensor-agnostic? |
|---|---|---|
| `BEVFormerEncoder` + `GeometryKernelAttention` (v1 default) | Project BEV reference points into cameras via `lidar2img` matrices; deformable cross-attention | **Yes** (geometry is analytical, not learned) |
| `BEVFormerEncoder` + `MSDeformableAttention3D` (v1 bevformer variant, v2) | Same as above | **Yes** |
| `LSSTransform` / `LSSTransformV2` (v1 bevpool variant, v2 default) | Lift-Splat-Shoot: learned depth distribution from image features + camera params | **No** (depth prior is learned from training-set camera distributions) |

This distinction is the root cause of most sensor-coupling issues.

---

## Part 1 — LSSTransform / DepthNet Couplings

These couplings are **specific to configs that use `LSSTransform` or
`LSSTransformV2`** as the BEV encoder (all MapTRv2 nuScenes / PandaSet configs,
and the v1 `_bevpool` variant).

### 1.1 DepthNet Camera-Parameter MLP — `Linear(22, …)` + `BatchNorm1d(22)`

**Severity: High (silent degradation on new sensor)**
**Files:** `projects/mmdet3d_plugin/maptr/modules/encoder.py:807–810, 1263–1279`

`get_mlp_input` assembles a 22-element vector per camera:

```
[fx, fy, cx, cy,                    # 4  intrinsics
 aug_rot(2×2), aug_trans(2),        # 6  image-augmentation params
 cam2ego_flat(3×4)]                 # 12 extrinsic (camera→ego)
= 22 elements
```

This vector is passed to `DepthNet`, which has `nn.Linear(22, mid_ch)` and
`nn.BatchNorm1d(22)` as entry layers.  The 22-element count is fixed in the
checkpoint.  More importantly, the BatchNorm running statistics and the MLP
weights encode a *learned prior over nuScenes camera parameter distributions*
(e.g., fx ≈ 1266, cx ≈ 800 for nuScenes).  A camera with substantially
different intrinsics (PandaSet fx ≈ 1350–1970) is out-of-distribution,
producing unreliable depth estimates → corrupted BEV features.

The same pattern is duplicated in:
- `LSSTransform.get_mlp_input` (line 1263)
- `LSSTransformV2.get_mlp_input` (line 1415)
- `BEVFormerEncoderDepth.get_mlp_input` (line 1040)

**Fix options:**
- Fine-tune `DepthNet` on the target dataset with LiDAR depth supervision
  (implemented in `tools/pandaset/finetune_depthnet.py`).
- Use LiDAR one-hot depth override at inference (`lss._lidar_depth`).
- Switch to a BEVFormer encoder that requires no depth estimation.

### 1.2 Depth Binning `dbound` → DepthNet Output-Head Size

**Severity: Critical (checkpoint incompatibility)**
**Files:** `encoder.py:72, 1086`; all LSS configs

```python
self.D = int((dbound[1] - dbound[0]) / dbound[2])   # encoder.py:72
self.depth_net = DepthNet(..., depth_channels=self.D, ...)  # encoder.py:1086
```

`D` is the output channel count of the final `Conv2d` in `DepthNet`.  With the
standard `dbound = [1.0, 35.0, 0.5]` this gives D = 68.  Changing `dbound`
changes `D`, which changes the weight shape → checkpoint loading fails with a
shape mismatch.  The depth range also constrains what scene geometry the model
can represent.

**Affected configs:** `maptrv2_nusc_r50_24ep.py:14`,
`maptrv2_nusc_r50_pandaset.py:18`, `maptr_tiny_r50_24e_bevpool.py:14`.

### 1.3 Frustum Cached from First Batch's `img_shape` (LSSTransform V1 Only)

**Severity: Medium (silent wrong geometry on resolution change)**
**File:** `encoder.py:81–83, 123–125`

In `LSSTransform` (V1), the pixel-to-ray frustum is created on the *first*
forward call using `img_metas[0]['img_shape'][0]` and then cached in
`self.frustum`:

```python
if self.frustum is None:
    self.frustum = self.create_frustum(fH, fW, img_metas)   # uses cam-0 H×W
```

Subsequent batches reuse this frustum even if the image resolution differs
(e.g., different test-time crop, mixed-resolution datasets).  `LSSTransformV2`
fixes this by computing the frustum once at `__init__` time from a config
`input_size` parameter.

### 1.4 Five Required `img_metas` Keys for LSS Forward

**Severity: Medium (crash on missing key)**
**File:** `encoder.py:253–275`

`BaseTransform.forward` unconditionally reads five keys from `img_metas`:
`lidar2img`, `camera2ego`, `camera_intrinsics`, `img_aug_matrix`, `lidar2ego`.
The `BEVFormerEncoder` path only requires `lidar2img`.  Datasets or pipelines
that do not populate all five will crash with a `KeyError`.

---

## Part 2 — BEVFormer / MSDeformableAttention Couplings

The BEVFormer encoder projects BEV reference points into camera images via
`lidar2img` matrices and samples features with deformable attention.  **All
camera geometry (focal length, principal point, extrinsics) is encoded in the
projection matrices, which are computed from calibration and passed as data —
not learned.**  This makes the core mechanism sensor-agnostic: swapping cameras
only requires providing correct `lidar2img` matrices.

The residual couplings listed below are not intrinsic to the attention
mechanism.

### 2.1 `cams_embeds` — Learned Per-Camera Identity Embedding

**Severity: High (checkpoint shape + semantic mismatch)**
**Files:** `maptr/modules/transformer.py:95–96, 188–189`;
`bevformer/modules/transformer.py:74–75`; `geometry_kernel_attention.py:35`;
`bevformer/modules/spatial_cross_attention.py:47`

```python
self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
# applied as:
feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
```

Two problems:
1. **Shape**: `num_cams` = 6 (default) is baked into the checkpoint.  A rig
   with a different camera count fails to load.
2. **Semantic**: Even when counts match, embedding row `i` was trained to encode
   nuScenes camera `i` (FRONT, FRONT_RIGHT, …).  A different rig with a
   different camera ordering will receive semantically wrong conditioning.

**Workaround:** Set `use_cams_embeds=False` in the encoder config.  The
attention then has no per-camera conditioning but becomes truly sensor-agnostic.
PandaSet configs already apply this workaround.

### 2.2 `point_sampling` Normalises All Cameras by Camera-0's Resolution

**Severity: Medium (silent wrong projection for multi-resolution rigs)**
**File:** `bevformer/modules/encoder.py:130–131`

```python
reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]   # cam-0 W
reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]   # cam-0 H
```

Uses the first camera's image dimensions to normalise projected coordinates for
*all* cameras.  For rigs where all cameras share the same padded resolution this
is harmless.  For rigs with different native resolutions (e.g., mix of fisheye
and pinhole cameras at different H×W before padding), cameras 1–N will have
incorrectly normalised coordinates.

---

## Part 3 — BEV Grid & Positional Encoding Couplings

These apply to all encoder backends.

### 3.1 `LearnedPositionalEncoding` Table Size = `bev_h × bev_w`

**Severity: Critical (checkpoint incompatibility)**
**Files:** all configs; `maptr_head.py:287`, `maptrv2_head.py:287`

```python
positional_encoding=dict(
    type='LearnedPositionalEncoding',
    row_num_embed=bev_h_,   # 200
    col_num_embed=bev_w_,)  # 100
```

The embedding tables have shape `(200, embed_dims)` and `(100, embed_dims)`,
baked into the checkpoint.  Changing `bev_h` or `bev_w` (e.g., to cover a
different `pc_range` at the same metric-per-cell resolution) requires retraining
from scratch.

### 3.2 `rotate_center=[100, 100]` Default Is Wrong for 200×100 BEV

**Severity: High (silent temporal misalignment — affects all datasets)**
**File:** `maptr/modules/transformer.py:63`; `bevformer/modules/transformer.py:50`

The previous-frame BEV is rotated by the ego heading delta to align it with the
current frame.  The rotation pivot `[100, 100]` is the centre of a *square*
200×200 grid.  The actual BEV is 200 (rows) × 100 (cols), whose true centre is
`[100, 50]`.  No config overrides this default.  The rotation is therefore
offset by 50 columns, producing a temporal misalignment artefact that worsens
with larger heading changes.  This is a bug, not a dataset coupling per se, but
it is amplified for datasets with wider roads (larger heading changes) or
different `bev_w`.

### 3.3 `post_center_range` Not Auto-Derived from `pc_range`

**Severity: Medium (silent prediction clipping)**
**File:** all configs (~line 163)

```python
post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
```

Detections whose decoded BEV coordinates fall outside this range are silently
discarded by `MapTRNMSFreeCoder`.  It must be updated manually when `pc_range`
changes.  Forgetting to do so silently clips valid far-range predictions.

---

## Part 4 — Ego-Motion & Coordinate System Couplings

### 4.1 LiDAR-Centric Ego Frame Convention

**Severity: High (silent rotation/mirror of entire BEV)**
**File:** `encoder.py:253–310`; `bevformer/modules/encoder.py:102–131`

The geometry pipeline assumes the LiDAR coordinate system is the ego frame:
X-forward, Y-left, Z-up (nuScenes convention).  `lidar2img` is expected to map
from this frame to image pixels.  If a dataset's annotation converter uses a
different LiDAR orientation (e.g., X-right, Y-forward), all BEV predictions
will be rotated 90° unless the converter explicitly remaps.

### 4.2 CAN Bus MLP — Fixed 18-Element Input, Elements 7–15 Unused

**Severity: Medium (wasted capacity; crash if vector is wrong size)**
**File:** `maptr/modules/transformer.py:99–106`

```python
self.can_bus_mlp = nn.Sequential(nn.Linear(18, embed_dims // 2), ...)
```

The 18-element layout:
- `[0:3]` — global XYZ translation (only `[0]`, `[1]` used as ego-motion delta)
- `[3:7]` — global quaternion
- `[7:15]` — physical CAN bus signals (acceleration, velocity, etc.) —
  set to `np.ones(9)` for non-nuScenes datasets; MLP treats them as data
- `[16]` — heading in radians
- `[17]` — heading in degrees

The MLP weight dimension (18) is baked into the checkpoint.  Non-nuScenes
datasets must pad to exactly 18 elements in this layout.  **Note:** the CAN bus
MLP is only invoked in the `attn_bev_encode` path (BEVFormer encoder).  In the
`lss_bev_encode` path the MLP is built but never called — `can_bus` is silently
dropped.

### 4.3 Temporal Ego-Delta Assumed in Global Frame (X=East, Y=North)

**Severity: Medium (wrong temporal shift direction)**
**File:** `maptr/modules/transformer.py:141–143`

```python
delta_x = np.array([each['can_bus'][0] for each in img_metas])
delta_y = np.array([each['can_bus'][1] for each in img_metas])
```

`can_bus[0:3]` is the ego2global translation.  The delta between frames is
taken in the global coordinate frame.  If the dataset's global CRS has different
axis orientation, the BEV shift will be in the wrong direction, producing
temporal BEV misalignment.

### 4.4 `scene_token` for Sequence Boundary Detection

**Severity: Low (stale BEV leaks across sequences if absent)**
**File:** `maptr/detectors/maptr.py:291–295`

```python
if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
    self.prev_frame_info['prev_bev'] = None
```

`scene_token` is a nuScenes concept.  Non-nuScenes datasets must supply this
field in their annotation pkl (using any unique per-sequence identifier) to
prevent stale temporal BEV from propagating across sequence boundaries.

---

## Part 5 — Map Class & Ground-Truth Couplings

### 5.1 `CLASS2LABEL` Hardcoded in Dataset Class

**Severity: Low (requires source edit to add classes)**
**File:** `projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py:493–498`

```python
CLASS2LABEL = {
    'road_divider': 0, 'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1,
}
```

This mapping is a class-level constant, not a config parameter.  Adding new map
element types (stop lines, speed bumps) requires editing source code.  The
number of classes also affects the classifier-head output dimension, which is
baked into the checkpoint.

### 5.2 `VectorizedLocalMap` Tied to NuScenesMap API

**Severity: High (dataset-locked GT generation)**
**File:** `nuscenes_map_dataset.py:518–528`

The ground-truth generator imports and uses the nuScenes Python SDK
(`NuScenesMap`) with four hardcoded geographic locations
(`boston-seaport`, `singapore-hollandvillage`, etc.).  This is only used during
annotation pre-processing, not in the network forward pass, but any dataset
using `CustomNuScenesLocalMapDataset` must have its map GT pre-rendered into
the pkl by a separate converter.

---

## Part 6 — v1-Specific Couplings (Not Present in v2)

### 6.1 GeometryKernelAttention — Compile-Time `kernel_size`

**Severity: Medium (recompile required to change)**
**Files:** `maptr/modules/geometry_kernel_attention.py:35`;
`maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn_cuda_kernel.cuh`

The `kernel_size=(3, 5)` parameter is compiled into the CUDA `.so` at build
time.  Changing the kernel shape requires rebuilding the extension.  v2 dropped
GKT in favour of standard `MSDeformableAttention3D`, eliminating this coupling.

### 6.2 `obtain_history_bev` Drops `lidar_feat` in Fusion Mode (v1 Bug)

**Severity: Medium (silent LiDAR feature loss in temporal path)**
**File:** `maptr/detectors/maptr.py:181`

In the multi-modal temporal path, the history BEV is computed by:
```python
prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
```
`lidar_feat` is not passed, so temporal frames silently lose LiDAR features
even when camera+LiDAR fusion is enabled.

### 6.3 `bev_encoder_type` String Match in v1 Head

**Severity: Low (silent `bev_embedding=None` for unrecognised encoder)**
**File:** `maptr/dense_heads/maptr_head.py:95, 177`

```python
self.bev_encoder_type = transformer.encoder.type
if self.bev_encoder_type == 'BEVFormerEncoder':
    self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
else:
    self.bev_embedding = None
```

Any encoder type whose string name is not exactly `'BEVFormerEncoder'` silently
disables BEV query embeddings.  A new encoder variant named e.g.
`'BEVFormerEncoderV2'` would fall through to `None` without a warning.

---

## Summary Table

| # | Coupling | Encoder path | Config / Brittle | Failure mode |
|---|---|---|---|---|
| 1.1 | DepthNet 22-elem mlp_input, OOD on new camera | LSS | Brittle (checkpoint BN stats) | Silent depth degradation |
| 1.2 | `dbound` → D baked into DepthNet output head | LSS | Brittle (weight shape) | Crash on checkpoint load |
| 1.3 | Frustum cached from first batch img_shape | LSS v1 | Brittle (runtime cache) | Wrong geometry on resolution change |
| 1.4 | 5 required img_meta keys | LSS | Brittle | Crash if key absent |
| 2.1 | `cams_embeds` shape = (num_cams, D) + semantic ordering | All | Brittle (shape) + Silent (semantic) | Crash (N≠6) or wrong conditioning |
| 2.2 | `point_sampling` uses cam-0 H×W for all cameras | BEVFormer | Silent | Wrong projection for mixed-res rigs |
| 3.1 | `LearnedPositionalEncoding` bev_h×bev_w baked in checkpoint | All | Brittle (weight shape) | Crash if BEV resolution changes |
| 3.2 | `rotate_center=[100,100]` wrong for 200×100 BEV | All | Brittle (bug) | Temporal BEV misalignment |
| 3.3 | `post_center_range` not derived from `pc_range` | All | Config (manual coupling) | Silent prediction clipping |
| 4.1 | LiDAR/ego frame convention (X-forward, Y-left, Z-up) | All | Convention | Silent BEV rotation/mirror |
| 4.2 | `can_bus_mlp Linear(18,…)`, elements 7–15 unused | BEVFormer | Brittle (weight shape) | Crash if can_bus ≠ 18 elems |
| 4.3 | Ego-delta in global frame axes | All | Convention | Wrong temporal shift direction |
| 4.4 | `scene_token` for sequence boundary | All | Data field | Stale temporal BEV across sequences |
| 5.1 | `CLASS2LABEL` hardcoded | All | Source edit required | Cannot add map classes from config |
| 5.2 | `VectorizedLocalMap` uses NuScenesMap API | All | Dataset-locked | Cannot generate GT outside nuScenes locations |
| 6.1 | GKT `kernel_size=(3,5)` compile-time | v1 GKT only | Brittle (compile-time) | Recompile to change kernel |
| 6.2 | `obtain_history_bev` drops `lidar_feat` | v1 fusion only | Bug | Silent LiDAR loss in temporal path |
| 6.3 | `bev_encoder_type` string match | v1 head only | Silent | `bev_embedding=None` for new encoders |

---

## Key Takeaway: BEVFormer + `use_cams_embeds=False` is Sensor-Agnostic

The `BEVFormerEncoder` + `MSDeformableAttention3D` path projects BEV reference
points into cameras via calibration-derived `lidar2img` matrices.  All sensor
geometry is handled analytically — no learned prior over focal length or
extrinsics.  With `use_cams_embeds=False` (to drop the per-camera identity
embedding), this path requires only:

- Correct `lidar2img` matrices in `img_metas` (from calibration)
- Consistent ego-frame convention
- A `scene_token` field for sequence boundaries

The `LSSTransform` path cannot be made sensor-agnostic by a config flag because
the DepthNet *learns a distribution over camera parameters*.  Cross-dataset
use requires either fine-tuning the DepthNet (see
`tools/pandaset/finetune_depthnet.py`) or replacing depth with a LiDAR-derived
override.
