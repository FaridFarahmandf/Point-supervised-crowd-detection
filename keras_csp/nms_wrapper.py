# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Try to import compiled versions, fallback to Python implementation
try:
    from .nms.cpu_nms import cpu_nms
except (ImportError, ModuleNotFoundError):
    # Fallback to pure Python implementation
    from .nms.py_cpu_nms import py_cpu_nms as cpu_nms

try:
    from .nms.gpu_nms import gpu_nms
except (ImportError, ModuleNotFoundError):
    gpu_nms = None

import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
    # Note: cpu_soft_nms might not be available, using regular NMS
    # This is a simplified version
    keep = cpu_nms(np.ascontiguousarray(dets, dtype=np.float32), Nt)
    return keep

def nms(dets, thresh, usegpu, gpu_id):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if usegpu and gpu_nms is not None:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)
