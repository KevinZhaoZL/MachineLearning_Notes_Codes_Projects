# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午10:56
# @Author  : Zhao Lei
# @File    : __init__.py.py
# @Desc    :

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback
