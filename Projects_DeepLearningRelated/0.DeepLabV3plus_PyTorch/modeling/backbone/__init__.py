# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午10:53
# @Author  : Zhao Lei
# @File    : __init__.py.py
# @Desc    :

from modeling.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
