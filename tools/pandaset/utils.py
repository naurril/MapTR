"""
Shared constants and helper functions for MapTR map-building scripts.
Dataset-agnostic: works with any mmdet3d config (nuScenes, PandaSet, AV2, …).
"""
import os.path as osp

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── constants ─────────────────────────────────────────────────────────────────

CLASS_COLORS = ['#FF8C00', '#1E90FF', '#32CD32']   # divider, ped_crossing, boundary
CLASS_NAMES  = ['divider', 'ped_crossing', 'boundary']

# Fallback PC range if not stored in the frame JSON (overridden at runtime)
_DEFAULT_PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]


# ── geometry ──────────────────────────────────────────────────────────────────

def to_world(pts_local, lidar2global):
    """
    pts_local    : (N, 2) float32 – x,y in local ego frame
    lidar2global : (4, 4) float64
    Returns (N, 2) world-frame x,y.
    """
    n = len(pts_local)
    h = np.ones((n, 4), dtype=np.float64)
    h[:, :2] = pts_local.astype(np.float64)
    h[:, 2]  = 0.0
    return (lidar2global @ h.T).T[:, :2]


def chamfer_dist_2d(a, b):
    """Bidirectional average Chamfer distance between two (N,2) arrays."""
    diff = a[:, None, :] - b[None, :, :]
    d2   = (diff ** 2).sum(-1)
    return 0.5 * (d2.min(1).mean() + d2.min(0).mean()) ** 0.5


def greedy_nms(candidates, nms_dist):
    """
    candidates : list of (score, pts_world)
    Returns the subset kept after greedy NMS (highest score first).
    """
    if not candidates:
        return []
    kept = []
    for score, pts in sorted(candidates, key=lambda x: x[0], reverse=True):
        if not any(chamfer_dist_2d(pts, kpts) < nms_dist for _, kpts in kept):
            kept.append((score, pts))
    return kept


def _pairwise_chamfer(pts_list):
    """Return symmetric N×N Chamfer distance matrix."""
    n = len(pts_list)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = chamfer_dist_2d(pts_list[i], pts_list[j])
            D[i, j] = D[j, i] = d
    return D


def _align_to_ref(pts, ref):
    """Return pts or pts reversed, whichever is closer to ref in Chamfer distance."""
    if chamfer_dist_2d(pts[::-1], ref) < chamfer_dist_2d(pts, ref):
        return pts[::-1]
    return pts


def _arc_params(pts):
    """Cumulative arc-length array and per-segment lengths for a polyline."""
    seglens = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # (P-1,)
    arc     = np.concatenate([[0.0], np.cumsum(seglens)])      # (P,)
    return seglens, arc


def _project_to_arc(pt, ref, seglens, arc):
    """
    Project pt onto the reference polyline, including endpoint tangent extensions.

    Returns (arc_coord, cross_signed):
      arc_coord    : arc-length position (may be <0 or >arc[-1] for extensions)
      cross_signed : signed perpendicular offset (positive = left of travel dir)
    """
    best_arc, best_cross, best_dist = 0.0, 0.0, np.inf

    for i in range(len(ref) - 1):
        sv = ref[i + 1] - ref[i]
        sl = seglens[i]
        if sl < 1e-10:
            continue
        t    = np.clip(np.dot(pt - ref[i], sv) / sl ** 2, 0.0, 1.0)
        proj = ref[i] + t * sv
        d    = np.linalg.norm(pt - proj)
        if d < best_dist:
            best_dist  = d
            best_arc   = arc[i] + t * sl
            perp       = np.array([-sv[1], sv[0]]) / sl
            best_cross = np.dot(pt - proj, perp)

    # Extension before start
    sv, sl = ref[1] - ref[0], seglens[0]
    if sl > 1e-10:
        t = np.dot(pt - ref[0], sv) / sl ** 2
        if t < 0:
            proj = ref[0] + t * sv
            d    = np.linalg.norm(pt - proj)
            if d < best_dist:
                best_dist  = d
                best_arc   = t * sl          # negative
                perp       = np.array([-sv[1], sv[0]]) / sl
                best_cross = np.dot(pt - proj, perp)

    # Extension after end
    sv, sl = ref[-1] - ref[-2], seglens[-1]
    if sl > 1e-10:
        t = np.dot(pt - ref[-2], sv) / sl ** 2
        if t > 1:
            proj = ref[-2] + t * sv
            d    = np.linalg.norm(pt - proj)
            if d < best_dist:
                best_arc   = arc[-2] + t * sl   # > arc[-1]
                perp       = np.array([-sv[1], sv[0]]) / sl
                best_cross = np.dot(pt - proj, perp)

    return best_arc, best_cross


def _eval_polyline(s, ref, seglens, arc):
    """
    Interpolate (or extrapolate along endpoint tangent) the reference polyline
    at arc-length s.  Returns (point, perp_unit_vector).
    """
    total = arc[-1]
    if s <= 0.0:
        sv   = ref[1] - ref[0]
        tang = sv / seglens[0]
        perp = np.array([-tang[1], tang[0]])
        return ref[0] + s * tang, perp
    if s >= total:
        sv   = ref[-1] - ref[-2]
        tang = sv / seglens[-1]
        perp = np.array([-tang[1], tang[0]])
        return ref[-1] + (s - total) * tang, perp
    idx  = int(np.clip(np.searchsorted(arc, s, side='right') - 1, 0, len(ref) - 2))
    sv   = ref[idx + 1] - ref[idx]
    sl   = seglens[idx]
    tang = sv / sl
    perp = np.array([-tang[1], tang[0]])
    return ref[idx] + (s - arc[idx]) / sl * sv, perp


def dbscan_best(candidates, nms_dist):
    """
    Cluster with DBSCAN (Chamfer distance metric), keep highest-score per cluster.

    candidates : list of (score, pts_world)
    nms_dist   : DBSCAN eps — same unit as Chamfer distance (metres)
    Returns one (score, pts) per cluster.
    """
    from sklearn.cluster import DBSCAN
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    pts_list = [pts for _, pts in candidates]
    D = _pairwise_chamfer(pts_list)
    labels = DBSCAN(eps=nms_dist, min_samples=1, metric='precomputed').fit_predict(D)

    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(candidates[idx])

    return [max(cluster, key=lambda x: x[0]) for cluster in clusters.values()]


def dbscan_weighted_merge(candidates, nms_dist):
    """
    Cluster with DBSCAN, then produce one score-weighted average polyline per cluster.

    Within each cluster, all polylines are aligned to the highest-score reference
    (reversed if that gives a smaller Chamfer distance) before averaging.

    candidates : list of (score, pts_world)
    nms_dist   : DBSCAN eps (metres)
    Returns one (score, pts) per cluster where pts is the weighted-average polyline.
    """
    from sklearn.cluster import DBSCAN
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    pts_list = [pts for _, pts in candidates]
    D = _pairwise_chamfer(pts_list)
    labels = DBSCAN(eps=nms_dist, min_samples=1, metric='precomputed').fit_predict(D)

    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(candidates[idx])

    merged = []
    for cluster in clusters.values():
        scores = np.array([s for s, _ in cluster])
        pts_arr = [pts for _, pts in cluster]
        ref = pts_arr[int(np.argmax(scores))]
        aligned = [_align_to_ref(pts, ref) for pts in pts_arr]
        weights = scores / scores.sum()
        avg_pts = sum(w * pts for w, pts in zip(weights, aligned))
        merged.append((float(scores.max()), avg_pts))
    return merged


def growing_merge(candidates, nms_dist, n_out=None, cluster_dist=None):
    """
    DBSCAN cluster + arc-length growing merge.

    Uses the highest-score member's polyline as the parametric backbone.
    Every point from every cluster member is projected onto that backbone
    (with endpoint tangent extension for points beyond its ends), giving
    an arc-length coordinate and a signed cross-track offset.  The UNION
    of all arc-length ranges is then re-sampled at n_out points using a
    Gaussian kernel — so the result grows longer as more frames contribute
    and handles straight, lateral, and curved polylines correctly.

    Closed polygons (e.g. ped_crossing, detected by first≈last point) fall
    back to the score-weighted average, which preserves their 2-D shape.

    candidates   : list of (score, pts_world)  where pts_world is (P, 2)
    nms_dist     : kept for API compatibility; used as cluster_dist fallback
    cluster_dist : DBSCAN eps (metres).  Should be tighter than nms_dist to
                   avoid merging distinct parallel elements (e.g. lane dividers
                   ~3 m apart).  Defaults to nms_dist / 2.
    n_out        : output points per polyline (default: same as input P)
    Returns one (score, pts) per cluster.
    """
    from sklearn.cluster import DBSCAN
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    eps      = cluster_dist if cluster_dist is not None else nms_dist / 2
    pts_list = [pts for _, pts in candidates]
    D        = _pairwise_chamfer(pts_list)
    labels   = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit_predict(D)

    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(candidates[idx])

    merged = []
    for cluster in clusters.values():
        scores  = np.array([s for s, _ in cluster])
        pts_arr = [pts for _, pts in cluster]
        P       = len(pts_arr[0])
        k       = n_out if n_out is not None else P
        ref     = pts_arr[int(np.argmax(scores))]
        aligned = [_align_to_ref(pts, ref) for pts in pts_arr]

        # Closed polygons: arc-length approach collapses them; use weighted avg.
        if np.linalg.norm(ref[0] - ref[-1]) < 1.0:
            w = scores / scores.sum()
            merged.append((float(scores.max()),
                           sum(wi * p for wi, p in zip(w, aligned))))
            continue

        seglens, arc = _arc_params(ref)
        if arc[-1] < 1e-3:
            w = scores / scores.sum()
            merged.append((float(scores.max()),
                           sum(wi * p for wi, p in zip(w, aligned))))
            continue

        # Project every point from every member onto the reference arc
        all_arc, all_cross, all_w = [], [], []
        for pts, sc in zip(aligned, scores):
            for pt in pts:
                a, c = _project_to_arc(pt, ref, seglens, arc)
                all_arc.append(a)
                all_cross.append(c)
                all_w.append(sc)
        all_arc   = np.array(all_arc)
        all_cross = np.array(all_cross)
        all_w     = np.array(all_w)

        # Score-weighted arc extent: only grow as far as positions where
        # accumulated score weight (Gaussian kernel sum) exceeds a threshold.
        # This prevents low-confidence outliers from spuriously extending ends.
        raw_min, raw_max = all_arc.min(), all_arc.max()
        probe_sigma = (arc[-1] / max(P - 1, 1)) * 1.5
        weight_thresh = all_w.max() * 0.05   # 5% of peak score
        n_probe = max(P * 4, 200)
        probe_s = np.linspace(raw_min, raw_max, n_probe)
        weight_at = np.array([
            (np.exp(-0.5 * ((all_arc - s) / probe_sigma) ** 2) * all_w).sum()
            for s in probe_s
        ])
        valid = weight_at >= weight_thresh
        if valid.any():
            arc_min = probe_s[valid][0]
            arc_max = probe_s[valid][-1]
        else:
            arc_min, arc_max = raw_min, raw_max

        # Auto n_out: keep the same inter-point spacing as the reference,
        # so longer merged elements get proportionally more points.
        if n_out is not None:
            k = n_out
        else:
            spacing = arc[-1] / max(P - 1, 1)
            k = max(P, int(round((arc_max - arc_min) / spacing)) + 1)

        sigma = (arc_max - arc_min) / max(k - 1, 1) * 1.5

        new_pts = []
        for s in np.linspace(arc_min, arc_max, k):
            kern  = np.exp(-0.5 * ((all_arc - s) / sigma) ** 2) * all_w
            tot   = kern.sum()
            cross = (kern * all_cross).sum() / tot if tot > 1e-10 else 0.0
            pt, perp = _eval_polyline(s, ref, seglens, arc)
            new_pts.append(pt + cross * perp)

        merged.append((float(scores.max()), np.array(new_pts)))
    return merged


# ── image helpers ─────────────────────────────────────────────────────────────

def make_surround_view(filenames, cols=3):
    """
    Build a surround-view mosaic from an ordered list of image paths.

    filenames : list of image file paths in camera order (as stored in the
                frame JSON / img_metas['filename']).  The order is determined
                by the dataset config — no dataset-specific name parsing is done.
    cols      : number of columns per row (default 3 for the standard 6-camera
                surround rig: front-left, front, front-right / back-left, …)

    Returns a numpy BGR image (HWC uint8).
    """
    imgs = []
    for p in filenames:
        img = cv2.imread(p) if (p and osp.isfile(p)) else np.zeros((270, 480, 3), dtype=np.uint8)
        imgs.append(img)

    if not imgs:
        return np.zeros((270, 480, 3), dtype=np.uint8)

    rows = []
    for i in range(0, len(imgs), cols):
        row_imgs = imgs[i:i + cols]
        h = min(im.shape[0] for im in row_imgs)
        row_imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                    for im in row_imgs]
        rows.append(cv2.hconcat(row_imgs))

    w = min(r.shape[1] for r in rows)
    return cv2.vconcat([r[:, :w] for r in rows])


def plot_bev(pts_local, labels, scores, out_path, pc_range=None, car_img=None):
    """
    Save a BEV prediction plot for a single frame.
    pts_local : (K, P, 2) in local ego frame  (already filtered by score thresh)
    labels    : (K,) int
    scores    : (K,) float
    """
    if pc_range is None:
        pc_range = _DEFAULT_PC_RANGE
    fig, ax = plt.subplots(figsize=(2, 4))
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.axis('off')
    for pts, label, score in zip(pts_local, labels, scores):
        color = CLASS_COLORS[int(label)] if int(label) < len(CLASS_COLORS) else 'gray'
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=1,       alpha=0.8, zorder=-1)
    if car_img is not None:
        ax.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    fig.savefig(out_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close(fig)


def render_global_map(elements, traj, title, out_path):
    """
    Render and save a global BEV map PNG.
    elements : list of (cls_idx, score, pts_world)  where pts_world is (P, 2)
    traj     : (T, 2) ego world positions
    """
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    ax.plot(traj[:, 0], traj[:, 1], color='white', linewidth=1.2, alpha=0.6, zorder=1)
    ax.scatter(traj[0, 0],  traj[0, 1],  color='white',  s=60, zorder=2, marker='o')
    ax.scatter(traj[-1, 0], traj[-1, 1], color='yellow', s=60, zorder=2, marker='*')

    for cls_idx, score, pts in elements:
        ax.plot(pts[:, 0], pts[:, 1],
                color=CLASS_COLORS[cls_idx],
                linewidth=1.0, alpha=min(1.0, score + 0.2), zorder=3)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    patches.append(mpatches.Patch(color='white', label='ego trajectory'))
    ax.legend(handles=patches, loc='upper right',
              facecolor='#2a2a4e', edgecolor='white', labelcolor='white', fontsize=9)
    ax.set_xlabel('World X (m)', color='white')
    ax.set_ylabel('World Y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.set_title(title, color='white', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
