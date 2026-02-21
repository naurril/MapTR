try:
    from .train import custom_train_model
    from .mmdet_train import custom_train_detector
except Exception:
    pass
from .test import custom_multi_gpu_test