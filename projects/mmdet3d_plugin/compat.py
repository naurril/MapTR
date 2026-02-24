"""
OpenMMLab v1 -> v2 compatibility shim.

Creates lazy proxy modules for old import paths (mmcv.runner.*, mmdet.core.*,
mmdet3d.core.*, etc.) that resolve to v2 equivalents on attribute access.
"""

import functools
import importlib
import sys
import types
import torch


class _LazyModule(types.ModuleType):
    """Module that lazily resolves attributes from a mapping."""

    def __init__(self, name, attr_map=None):
        super().__init__(name)
        self._attr_map = attr_map or {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._attr_map:
            source_mod, source_attr = self._attr_map[name]
            mod = importlib.import_module(source_mod)
            val = getattr(mod, source_attr)
            setattr(self, name, val)  # cache
            return val
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


def _ensure_module(dotted_name):
    """Create stub parent modules, never overriding real importable packages."""
    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if prefix not in sys.modules:
            try:
                importlib.import_module(prefix)
            except ImportError:
                sys.modules[prefix] = types.ModuleType(prefix)


def _install_lazy(dotted_name, attr_map):
    """Install a lazy module at dotted_name."""
    _ensure_module(dotted_name)
    if dotted_name not in sys.modules:
        sys.modules[dotted_name] = _LazyModule(dotted_name, attr_map)
    else:
        mod = sys.modules[dotted_name]
        if isinstance(mod, _LazyModule):
            mod._attr_map.update(attr_map)
        else:
            # Real module — add lazy fallback via instance __getattr__
            def make_getattr(m, amap):
                def __getattr__(name):
                    if name in amap:
                        src_mod, src_attr = amap[name]
                        real = importlib.import_module(src_mod)
                        val = getattr(real, src_attr)
                        setattr(m, name, val)
                        return val
                    raise AttributeError(
                        f"module {m.__name__!r} has no attribute {name!r}")
                return __getattr__
            # Merge with existing __getattr__ if any
            existing = getattr(mod, '__getattr__', None)
            if existing and callable(existing):
                inner = make_getattr(mod, attr_map)
                def chained(name, _inner=inner, _existing=existing):
                    try:
                        return _inner(name)
                    except AttributeError:
                        return _existing(name)
                mod.__getattr__ = chained
            else:
                mod.__getattr__ = make_getattr(mod, attr_map)


# ──────────────────────────────────────────────────────────────
# Stubs for removed APIs
# ──────────────────────────────────────────────────────────────

def force_fp32(apply_to=None, out_fp16=False):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def auto_fp16(apply_to=None, out_fp32=False):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


class _EvalHookStub:
    def __init__(self, *args, **kwargs):
        pass


def _get_host_info():
    import socket
    return socket.gethostname()


def _get_root_logger(log_file=None, log_level=None):
    from mmengine.logging import MMLogger
    try:
        return MMLogger.get_current_instance()
    except Exception:
        return MMLogger.get_instance("mmdet3d", log_level=log_level or "INFO")


class DataContainer:
    """Minimal stub for removed mmcv DataContainer."""
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                 pad_dims=2):
        self._data = data
        self.datatype = type(data)
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        self._pad_dims = pad_dims

    @property
    def data(self):
        return self._data

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    def __getitem__(self, index):
        return DataContainer(
            self._data[index],
            stack=self._stack,
            padding_value=self._padding_value,
            cpu_only=self._cpu_only,
            pad_dims=self._pad_dims,
        )

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'


class DefaultFormatBundle:
    """Minimal DefaultFormatBundle stub from old mmdet."""
    def __init__(self):
        pass

    def __call__(self, results):
        import torch
        import numpy as np
        if 'img' in results:
            img = results['img']
            if isinstance(img, list):
                # multi-view: list of HxWxC arrays -> stack into NxCxHxW
                imgs = []
                for im in img:
                    if len(im.shape) < 3:
                        im = np.expand_dims(im, -1)
                    imgs.append(np.ascontiguousarray(im.transpose(2, 0, 1)))
                img_tensor = torch.stack([torch.from_numpy(i) for i in imgs])
                results['img'] = DataContainer(img_tensor, stack=True)
            else:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results['img'] = DataContainer(
                    torch.from_numpy(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DataContainer(_to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DataContainer(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DataContainer(
                _to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results


class DefaultFormatBundle3D(DefaultFormatBundle):
    """DefaultFormatBundle3D ported from old mmdet3d for compatibility."""
    def __init__(self, class_names, with_gt=True, with_label=True):
        super().__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        from mmdet3d.structures import BasePoints
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DataContainer(results['points'].tensor)
        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DataContainer(_to_tensor(results[key]), stack=False)
        if self.with_gt:
            if 'gt_bboxes_3d_mask' in results:
                m = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][m]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][m]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][m]
                if 'depths' in results:
                    results['depths'] = results['depths'][m]
            if 'gt_bboxes_mask' in results:
                m = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][m]
                results['gt_names'] = results['gt_names'][m]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(results['gt_names'][0], list):
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res], dtype=np.int64)
                        for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array(
                        [self.class_names.index(n) for n in results['gt_names']], dtype=np.int64)
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array(
                        [self.class_names.index(n) for n in results['gt_names_3d']], dtype=np.int64)
        results = super().__call__(results)
        return results


def _to_tensor(data):
    """Convert objects of various python types to torch.Tensor."""
    import torch
    import numpy as np
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def _mmcv_jit_stub(derivate=False, coderize=False):
    """No-op stub for removed mmcv.jit decorator."""
    def decorator(fn):
        return fn
    return decorator


def _collate(batch, samples_per_gpu=1):
    """Replacement for mmcv.parallel.collate that handles DataContainer objects."""
    import collections.abc
    if not isinstance(batch, collections.abc.Sequence):
        raise TypeError(f'{type(batch)} is not supported.')

    # If first element is DataContainer, handle specially
    if isinstance(batch[0], DataContainer):
        stacked = []
        for dc in batch:
            stacked.append(dc.data)
        if batch[0].stack:
            # Stack tensors
            import torch as _torch
            if isinstance(stacked[0], _torch.Tensor):
                return DataContainer(
                    _torch.stack(stacked, dim=0),
                    stack=batch[0].stack,
                    padding_value=batch[0].padding_value,
                    cpu_only=batch[0].cpu_only,
                )
        # Otherwise just collect as list
        return DataContainer(
            stacked,
            stack=batch[0].stack,
            padding_value=batch[0].padding_value,
            cpu_only=batch[0].cpu_only,
        )
    elif isinstance(batch[0], dict):
        result = {}
        for key in batch[0]:
            result[key] = _collate([d[key] for d in batch], samples_per_gpu)
        return result
    elif isinstance(batch[0], (list, tuple)):
        transposed = list(zip(*batch))
        return [_collate(s, samples_per_gpu) for s in transposed]
    else:
        # fallback: use torch default_collate
        try:
            from torch.utils.data.dataloader import default_collate as _dc
            return _dc(batch)
        except Exception:
            return batch


def _scatter_val(val, device):
    """Recursively scatter a value: unwrap DataContainers, move tensors to device."""
    if isinstance(val, DataContainer):
        inner = val.data
        if not val.cpu_only and isinstance(inner, torch.Tensor):
            return inner.to(device)
        # For cpu_only (e.g. img_metas): return the raw data as-is
        return inner
    elif isinstance(val, list):
        return [_scatter_val(v, device) for v in val]
    elif isinstance(val, torch.Tensor):
        return val.to(device)
    return val


class _MMDataParallel(torch.nn.Module):
    """Single-GPU replacement for old mmcv.parallel.MMDataParallel."""
    def __init__(self, module, device_ids=None, **kwargs):
        super().__init__()
        self.module = module
        if device_ids:
            self.module = self.module.cuda(device_ids[0])

    def forward(self, *args, **kwargs):
        try:
            device = next(self.module.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scattered = {k: _scatter_val(v, device) for k, v in kwargs.items()}
        return self.module(*args, **scattered)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _bev_pool_stub(feats, coords, B, D, H, W):
    """Pure-PyTorch fallback for bev_pool (no CUDA extension required)."""
    import torch
    assert feats.shape[0] == coords.shape[0]
    ranks = (coords[:, 0] * (W * D * B)
             + coords[:, 1] * (D * B)
             + coords[:, 2] * B
             + coords[:, 3])
    indices = ranks.argsort()
    feats = feats[indices]
    coords = coords[indices]
    ranks = ranks[indices]
    # cumulative-sum pool (pure Python, no CUDA ext)
    feats = feats.float().cumsum(0)
    kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    feats = feats[kept]
    coords = coords[kept]
    feats = torch.cat((feats[:1], feats[1:] - feats[:-1]))
    B, D, H, W = int(B), int(D), int(H), int(W)
    final = torch.zeros((B, D, H, W, feats.shape[1]),
                        device=feats.device, dtype=feats.dtype)
    coords = coords.long()
    final[coords[:, 3], coords[:, 2], coords[:, 0], coords[:, 1]] = feats
    return final.permute(0, 4, 1, 2, 3).contiguous()


def _build_loss(cfg):
    # Plugin losses register in mmdet3d.registry.MODELS (via compat LOSSES mapping).
    # Use mmdet3d registry which is a child of mmengine and contains plugin classes.
    from mmdet3d.registry import MODELS
    return MODELS.build(cfg)


def _build_dataset(cfg, default_args=None):
    _register_transform_aliases()  # ensure aliases exist before pipeline is built
    from mmdet3d.registry import DATASETS as DS3D
    from mmdet.registry import DATASETS as DSDET
    # Try mmdet3d first, then mmdet (plugin classes register into mmdet)
    type_name = cfg.get('type', '') if hasattr(cfg, 'get') else cfg['type']
    if type_name in DS3D:
        return DS3D.build(cfg)
    return DSDET.build(cfg)


def _build_model(cfg, train_cfg=None, test_cfg=None):
    from mmdet3d.registry import MODELS as MODELS3D
    from mmdet.registry import MODELS as MODELSDET
    if train_cfg is not None:
        cfg['train_cfg'] = train_cfg
    if test_cfg is not None:
        cfg['test_cfg'] = test_cfg
    type_name = cfg.get('type', '') if hasattr(cfg, 'get') else cfg['type']
    if type_name in MODELS3D:
        return MODELS3D.build(cfg)
    return MODELSDET.build(cfg)


def _train_detector_stub(model, dataset, cfg, **kwargs):
    raise NotImplementedError(
        "train_detector is removed in OpenMMLab v2. Use mmengine.runner.Runner instead.")


def _wrap_fp16_model(model):
    """No-op: fp16 is handled by autocast in PyTorch >=1.6."""
    return model


def _set_random_seed(seed, deterministic=False):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _replace_ImageToTensor(pipelines):
    """Replace ImageToTensor with DefaultFormatBundle for batch_size > 1.
    In v2 this transform no longer exists; return pipeline unchanged."""
    return pipelines


def _single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Minimal single-GPU test loop compatible with old mmdet3d API."""
    import torch
    from mmengine.structures import InstanceData
    model.eval()
    results = []
    import mmcv
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset)) if hasattr(mmcv, 'ProgressBar') else None
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        if prog_bar:
            prog_bar.update()
    return results


_COMPAT_MOD = "projects.mmdet3d_plugin.compat"


def install():
    sys.modules.setdefault(_COMPAT_MOD, sys.modules[__name__])

    # ── mmcv top-level additions ──────────────────────────────
    _install_lazy("mmcv", {
        "DictAction": ("mmengine.config", "DictAction"),
        "mkdir_or_exist": ("mmengine.utils", "mkdir_or_exist"),
        "ProgressBar": ("mmengine.utils", "ProgressBar"),
        "symlink": ("mmengine.utils", "symlink"),
        "scandir": ("mmengine.utils", "scandir"),
    })

    # ── mmcv.runner ──────────────────────────────────────────
    _install_lazy("mmcv.runner", {
        "BaseModule": ("mmengine.model", "BaseModule"),
        "ModuleList": ("mmengine.model", "ModuleList"),
        "Sequential": ("mmengine.model", "Sequential"),
        "force_fp32": (_COMPAT_MOD, "force_fp32"),
        "auto_fp16": (_COMPAT_MOD, "auto_fp16"),
        "get_dist_info": ("mmengine.dist", "get_dist_info"),
        "init_dist": ("mmengine.dist", "init_dist"),
        "load_checkpoint": ("mmengine.runner", "load_checkpoint"),
        "wrap_fp16_model": (_COMPAT_MOD, "_wrap_fp16_model"),
        "HOOKS": ("mmengine.registry", "HOOKS"),
        "DistSamplerSeedHook": ("mmengine.hooks", "DistSamplerSeedHook"),
        "EpochBasedRunner": ("mmengine.runner", "Runner"),
        "Fp16OptimizerHook": ("mmengine.optim", "AmpOptimWrapper"),
        "OptimizerHook": ("mmengine.optim", "OptimWrapper"),
        "build_optimizer": ("mmengine.optim", "build_optim_wrapper"),
        "build_runner": ("mmengine.runner", "Runner"),
        "_load_checkpoint": ("mmengine.runner.checkpoint", "_load_checkpoint"),
        "HOOKS": ("mmengine.registry", "HOOKS"),
        "Hook": ("mmengine.hooks", "Hook"),
        "DistSamplerSeedHook": ("mmengine.hooks", "DistSamplerSeedHook"),
        "EpochBasedRunner": ("mmengine.runner", "Runner"),
        "EvalHook": (_COMPAT_MOD, "_EvalHookStub"),
        "DistEvalHook": (_COMPAT_MOD, "_EvalHookStub"),
        "save_checkpoint": ("mmengine.runner", "save_checkpoint"),
    })

    _install_lazy("mmcv.runner.base_module", {
        "BaseModule": ("mmengine.model", "BaseModule"),
        "ModuleList": ("mmengine.model", "ModuleList"),
        "Sequential": ("mmengine.model", "Sequential"),
    })

    _install_lazy("mmcv.runner.hooks.hook", {
        "HOOKS": ("mmengine.registry", "HOOKS"),
        "Hook": ("mmengine.hooks", "Hook"),
    })

    _install_lazy("mmcv.runner.hooks", {
        "HOOKS": ("mmengine.registry", "HOOKS"),
        "Hook": ("mmengine.hooks", "Hook"),
        "DistSamplerSeedHook": ("mmengine.hooks", "DistSamplerSeedHook"),
    })

    _install_lazy("mmcv.runner.optimizer.builder", {
        "OPTIMIZERS": ("mmengine.registry", "OPTIMIZERS"),
    })

    _install_lazy("mmcv.runner.epoch_based_runner", {
        "EpochBasedRunner": ("mmengine.runner", "Runner"),
    })

    _install_lazy("mmcv.runner.base_runner", {
        "BaseRunner": ("mmengine.runner", "Runner"),
    })

    _install_lazy("mmcv.runner.builder", {
        "RUNNERS": ("mmengine.registry", "RUNNERS"),
    })

    _install_lazy("mmcv.runner.utils", {
        "get_host_info": (_COMPAT_MOD, "_get_host_info"),
    })

    _install_lazy("mmcv.runner.checkpoint", {
        "save_checkpoint": ("mmengine.runner", "save_checkpoint"),
    })

    # ── mmcv.parallel (DataContainer, MMDataParallel) ────────
    _install_lazy("mmcv.parallel", {
        "DataContainer": (_COMPAT_MOD, "DataContainer"),
        "MMDataParallel": (_COMPAT_MOD, "_MMDataParallel"),
        "MMDistributedDataParallel": ("mmengine.model", "MMDistributedDataParallel"),
        "collate": (_COMPAT_MOD, "_collate"),
    })

    _install_lazy("mmcv.parallel.data_container", {
        "DataContainer": (_COMPAT_MOD, "DataContainer"),
    })

    # ── mmcv.utils ───────────────────────────────────────────
    _install_lazy("mmcv.utils", {
        "build_from_cfg": ("mmengine.registry", "build_from_cfg"),
        "ConfigDict": ("mmengine.config", "ConfigDict"),
        "deprecated_api_warning": ("mmengine.utils", "deprecated_api_warning"),
        "digit_version": ("mmengine.utils", "digit_version"),
        "print_log": ("mmengine.logging", "print_log"),
        "TORCH_VERSION": ("mmengine.utils.dl_utils", "TORCH_VERSION"),
        "Registry": ("mmengine.registry", "Registry"),
        "ext_loader": ("mmcv.utils.ext_loader", "ext_loader"),
        "to_2tuple": ("mmengine.utils", "to_2tuple"),
    })

    _install_lazy("mmcv.utils.registry", {
        "Registry": ("mmengine.registry", "Registry"),
        "build_from_cfg": ("mmengine.registry", "build_from_cfg"),
    })

    # ── mmcv top-level ─────────────────────────────────────────
    _install_lazy("mmcv", {
        "ConfigDict": ("mmengine.config", "ConfigDict"),
        "deprecated_api_warning": ("mmengine.utils", "deprecated_api_warning"),
        "Config": ("mmengine.config", "Config"),
        "jit": (_COMPAT_MOD, "_mmcv_jit_stub"),
    })

    # ── mmcv.image ───────────────────────────────────────────
    _install_lazy("mmcv.image", {
        "tensor2imgs": ("mmcv.image.misc", "tensor2imgs"),
    })

    # ── mmseg.ops ────────────────────────────────────────────
    _install_lazy("mmseg.ops", {
        "resize": ("mmseg.models.utils.wrappers", "resize"),
    })

    # ── mmcv.cnn patches ──────────────────────────────────────
    _install_lazy("mmcv.cnn", {
        "trunc_normal_init": ("mmengine.model.weight_init", "trunc_normal_init"),
        "bias_init_with_prob": ("mmengine.model.weight_init", "bias_init_with_prob"),
        "xavier_init": ("mmengine.model.weight_init", "xavier_init"),
        "constant_init": ("mmengine.model.weight_init", "constant_init"),
    })

    _install_lazy("mmcv.cnn.utils.weight_init", {
        "constant_init": ("mmengine.model.weight_init", "constant_init"),
    })

    # ── mmcv.cnn.bricks.registry ─────────────────────────────
    # ATTENTION etc. registries → MODELS in mmengine
    _install_lazy("mmcv.cnn.bricks.registry", {
        "ATTENTION": ("mmengine.registry", "MODELS"),
        "FEEDFORWARD_NETWORK": ("mmengine.registry", "MODELS"),
        "POSITIONAL_ENCODING": ("mmengine.registry", "MODELS"),
        "TRANSFORMER_LAYER": ("mmengine.registry", "MODELS"),
        "TRANSFORMER_LAYER_SEQUENCE": ("mmengine.registry", "MODELS"),
    })

    # ── mmdet.core ───────────────────────────────────────────
    _install_lazy("mmdet.core", {
        "multi_apply": ("mmdet.models.utils", "multi_apply"),
        "reduce_mean": ("mmdet.utils", "reduce_mean"),
        "EvalHook": (_COMPAT_MOD, "_EvalHookStub"),
        "encode_mask_results": ("mmdet.structures.mask", "encode_mask_results"),
    })

    _install_lazy("mmdet.core.bbox", {
        "BaseBBoxCoder": ("mmdet.models.task_modules.coders", "BaseBBoxCoder"),
    })

    _install_lazy("mmdet.core.bbox.builder", {
        "BBOX_ASSIGNERS": ("mmengine.registry", "TASK_UTILS"),
        "BBOX_CODERS": ("mmengine.registry", "TASK_UTILS"),
    })

    _install_lazy("mmdet.core.bbox.assigners", {
        "AssignResult": ("mmdet.models.task_modules.assigners", "AssignResult"),
        "BaseAssigner": ("mmdet.models.task_modules.assigners", "BaseAssigner"),
    })

    _install_lazy("mmdet.core.bbox.match_costs", {
        "build_match_cost": ("mmdet.models.task_modules.builder", "build_match_cost"),
    })

    _install_lazy("mmdet.core.bbox.match_costs.builder", {
        "MATCH_COST": ("mmengine.registry", "TASK_UTILS"),
    })

    _install_lazy("mmdet.core.bbox.transforms", {
        "bbox_xyxy_to_cxcywh": ("mmdet.structures.bbox", "bbox_xyxy_to_cxcywh"),
        "bbox_cxcywh_to_xyxy": ("mmdet.structures.bbox", "bbox_cxcywh_to_xyxy"),
    })

    _install_lazy("mmdet.core.evaluation.bbox_overlaps", {
        "bbox_overlaps": ("mmdet.structures.bbox", "bbox_overlaps"),
    })

    _install_lazy("mmdet.core.evaluation.eval_hooks", {
        "DistEvalHook": (_COMPAT_MOD, "_EvalHookStub"),
    })

    # ── mmdet registries (v1 had them under mmdet.models/datasets) ──
    # Use mmdet3d.registry.MODELS so MVXTwoStageDetector can resolve sub-modules
    _install_lazy("mmdet.models", {
        "DETECTORS": ("mmdet3d.registry", "MODELS"),
        "HEADS": ("mmdet3d.registry", "MODELS"),
        "LOSSES": ("mmdet3d.registry", "MODELS"),
        "NECKS": ("mmdet3d.registry", "MODELS"),
        "ROI_EXTRACTORS": ("mmdet3d.registry", "MODELS"),
        "BACKBONES": ("mmdet3d.registry", "MODELS"),
        "build_loss": (_COMPAT_MOD, "_build_loss"),
    })

    _install_lazy("mmdet.models.builder", {
        "BACKBONES": ("mmdet3d.registry", "MODELS"),
        "DETECTORS": ("mmdet3d.registry", "MODELS"),
        "HEADS": ("mmdet3d.registry", "MODELS"),
        "LOSSES": ("mmdet3d.registry", "MODELS"),
        "NECKS": ("mmdet3d.registry", "MODELS"),
        "build_loss": (_COMPAT_MOD, "_build_loss"),
    })

    _install_lazy("mmdet.models.utils.builder", {
        "TRANSFORMER": ("mmdet3d.registry", "MODELS"),
    })

    _install_lazy("mmdet.models.utils.transformer", {
        "inverse_sigmoid": ("mmdet.models.layers", "inverse_sigmoid"),
    })

    _install_lazy("mmdet.datasets", {
        "DATASETS": ("mmdet.registry", "DATASETS"),
        "CocoDataset": ("mmdet.datasets.coco", "CocoDataset"),
    })

    _install_lazy("mmdet.datasets.builder", {
        # Use root mmengine TRANSFORMS so BaseDataset.pipeline can find custom ops
        "PIPELINES": ("mmengine.registry", "TRANSFORMS"),
    })

    _install_lazy("mmdet.datasets.pipelines", {
        "to_tensor": (_COMPAT_MOD, "_to_tensor"),
        "LoadAnnotations": ("mmdet.datasets.transforms", "LoadAnnotations"),
        "LoadImageFromFile": ("mmcv.transforms", "LoadImageFromFile"),
    })

    # ── mmdet.utils ──────────────────────────────────────────
    _install_lazy("mmdet.utils", {
        "get_root_logger": (_COMPAT_MOD, "_get_root_logger"),
    })

    # ── mmdet.apis ───────────────────────────────────────────
    _install_lazy("mmdet.apis", {
        "train_detector": (_COMPAT_MOD, "_train_detector_stub"),
        "set_random_seed": (_COMPAT_MOD, "_set_random_seed"),
    })

    _install_lazy("mmdet.datasets", {
        "replace_ImageToTensor": (_COMPAT_MOD, "_replace_ImageToTensor"),
        "build_dataset": (_COMPAT_MOD, "_build_dataset"),
    })

    _install_lazy("mmdet.utils", {
        "get_root_logger": (_COMPAT_MOD, "_get_root_logger"),
    })

    # ── mmdet3d.core ─────────────────────────────────────────
    _install_lazy("mmdet3d.core", {
        "bbox3d2result": ("mmdet3d.structures", "bbox3d2result"),
    })

    _install_lazy("mmdet3d.core.bbox", {
        "BaseInstance3DBoxes": ("mmdet3d.structures", "BaseInstance3DBoxes"),
        "CameraInstance3DBoxes": ("mmdet3d.structures", "CameraInstance3DBoxes"),
        "get_box_type": ("mmdet3d.structures", "get_box_type"),
    })

    _install_lazy("mmdet3d.core.bbox.coders", {
        "build_bbox_coder": ("mmdet3d.models.task_modules.builder", "build_bbox_coder"),
    })

    _install_lazy("mmdet3d.core.bbox.iou_calculators", {
        "BboxOverlaps3D": ("mmdet3d.structures.ops", "BboxOverlaps3D"),
    })

    _install_lazy("mmdet3d.core.points", {
        "BasePoints": ("mmdet3d.structures", "BasePoints"),
        "get_points_type": ("mmdet3d.structures.points", "get_points_type"),
    })

    # ── mmdet3d registries/utils ─────────────────────────────
    _install_lazy("mmdet3d.models.detectors.mvx_two_stage", {
        "MVXTwoStageDetector": ("mmdet3d.models.detectors", "MVXTwoStageDetector"),
    })

    _install_lazy("mmdet3d.utils", {
        "get_root_logger": (_COMPAT_MOD, "_get_root_logger"),
        "collect_env": ("mmengine.utils.dl_utils", "collect_env"),
    })

    _install_lazy("mmdet3d.apis", {
        "single_gpu_test": (_COMPAT_MOD, "_single_gpu_test"),
    })

    _install_lazy("mmdet3d.datasets", {
        "build_dataset": (_COMPAT_MOD, "_build_dataset"),
    })

    _install_lazy("mmdet3d.models", {
        "build_model": (_COMPAT_MOD, "_build_model"),
    })

    _install_lazy("mmdet3d.datasets", {
        "NuScenesDataset": ("mmdet3d.datasets.nuscenes_dataset", "NuScenesDataset"),
    })

    # ── mmdet3d.ops (removed in v2, classes renamed) ──
    _install_lazy("mmdet3d.ops", {
        "Voxelization": ("mmdet3d.models.data_preprocessors.voxelize", "VoxelizationByGridShape"),
        "DynamicScatter": ("mmdet3d.models.data_preprocessors.voxelize", "DynamicScatter3D"),
        "bev_pool": (_COMPAT_MOD, "_bev_pool_stub"),
    })

    # ── mmdet3d.models.builder ──
    _install_lazy("mmdet3d.models.builder", {
        "build_backbone": (_COMPAT_MOD, "_build_loss"),
        "build_neck": (_COMPAT_MOD, "_build_loss"),
        "build_head": (_COMPAT_MOD, "_build_loss"),
        "build_detector": (_COMPAT_MOD, "_build_loss"),
        "build_fusion_layer": (_COMPAT_MOD, "_build_loss"),
        "build_voxel_encoder": (_COMPAT_MOD, "_build_loss"),
        "build_middle_encoder": (_COMPAT_MOD, "_build_loss"),
    })

    # ── mmdet3d.datasets.pipelines (removed in v2) ──
    _install_lazy("mmdet3d.datasets.pipelines", {
        "DefaultFormatBundle3D": (_COMPAT_MOD, "DefaultFormatBundle3D"),
    })


def _register_model_aliases():
    """Register mmcv v1 transformer layer class aliases into mmengine.MODELS."""
    from mmengine.registry import MODELS
    # In mmcv v1, DetrTransformerDecoderLayer was an alias for BaseTransformerLayer.
    # In mmcv v2 it's no longer registered, but configs still use the old name.
    if 'DetrTransformerDecoderLayer' not in MODELS:
        from mmcv.cnn.bricks.transformer import BaseTransformerLayer
        MODELS.register_module(name='DetrTransformerDecoderLayer', force=True)(BaseTransformerLayer)
    # CustomBaseTransformerLayer -> BaseTransformerLayer alias
    if 'CustomBaseTransformerLayer' not in MODELS:
        from mmcv.cnn.bricks.transformer import BaseTransformerLayer
        MODELS.register_module(name='CustomBaseTransformerLayer', force=True)(BaseTransformerLayer)
    # Positional encodings: in mmdet v3 they register in mmdet.registry.MODELS (a child),
    # but build_positional_encoding uses mmengine.registry.MODELS (root) which doesn't
    # search children — so pull them up explicitly.
    if 'LearnedPositionalEncoding' not in MODELS or 'SinePositionalEncoding' not in MODELS:
        try:
            from mmdet.models.layers.positional_encoding import (
                LearnedPositionalEncoding, SinePositionalEncoding)
            if 'LearnedPositionalEncoding' not in MODELS:
                MODELS.register_module(name='LearnedPositionalEncoding', force=True)(LearnedPositionalEncoding)
            if 'SinePositionalEncoding' not in MODELS:
                MODELS.register_module(name='SinePositionalEncoding', force=True)(SinePositionalEncoding)
        except Exception:
            pass
    # PseudoSampler: in mmdet v3 it's in mmdet.registry.TASK_UTILS (child), but
    # code uses mmengine.registry.TASK_UTILS (root) which doesn't search children.
    from mmengine.registry import TASK_UTILS
    if 'PseudoSampler' not in TASK_UTILS:
        try:
            from mmdet.models.task_modules.samplers import PseudoSampler
            TASK_UTILS.register_module(name='PseudoSampler', force=True)(PseudoSampler)
        except Exception:
            pass

    # MVXTwoStageDetector (mmdet3d v2) uses mmdet3d.registry.MODELS to build all
    # sub-components including img_backbone / img_neck which are mmdet models.
    # mmdet.registry.MODELS is a sibling (not child) of mmdet3d.registry.MODELS,
    # so we must mirror the needed mmdet model classes into mmdet3d.registry.MODELS.
    from mmdet3d.registry import MODELS as M3D
    try:
        import mmdet.models  # trigger full registration into mmdet.registry.MODELS
        from mmdet.registry import MODELS as MDMODELS
        for _name, _cls in list(MDMODELS._module_dict.items()):
            if _name not in M3D._module_dict:
                try:
                    M3D.register_module(name=_name, force=False)(_cls)
                except Exception:
                    pass
    except Exception:
        pass


def _register_transform_aliases():
    """Register name aliases and missing transforms into mmengine.TRANSFORMS."""
    from mmengine.registry import TRANSFORMS

    # Alias Custom* → canonical names used in configs
    _aliases = {
        'LoadMultiViewImageFromFiles': 'CustomLoadMultiViewImageFromFiles',
        'DefaultFormatBundle3D': 'CustomDefaultFormatBundle3D',
    }
    for alias, src in _aliases.items():
        if src in TRANSFORMS and alias not in TRANSFORMS:
            TRANSFORMS.register_module(name=alias, force=True)(TRANSFORMS.get(src))

    # Pull mmdet3d standard transforms into mmengine TRANSFORMS root
    _mmdet3d_transforms = [
        ('mmdet3d.datasets.transforms', 'LoadAnnotations3D'),
        ('mmdet3d.datasets.transforms', 'ObjectRangeFilter'),
        ('mmdet3d.datasets.transforms', 'ObjectNameFilter'),
        ('mmdet3d.datasets.transforms', 'GlobalRotScaleTrans'),
        ('mmdet3d.datasets.transforms', 'RandomFlip3D'),
        ('mmdet3d.datasets.transforms', 'PointsRangeFilter'),
        ('mmdet3d.datasets.transforms', 'PointShuffle'),
        ('mmdet3d.datasets.transforms', 'LoadPointsFromFile'),
        ('mmdet3d.datasets.transforms', 'LoadPointsFromMultiSweeps'),
        ('mmdet3d.datasets.transforms', 'MultiScaleFlipAug3D'),
    ]
    for mod_name, cls_name in _mmdet3d_transforms:
        if cls_name not in TRANSFORMS:
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                TRANSFORMS.register_module(name=cls_name, force=True)(cls)
            except Exception:
                pass


# Auto-install on import
install()
_register_model_aliases()
