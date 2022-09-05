import numpy as np
# import cv2

import torch.nn.functional as F


def post_get_top_20(layout_map, flag_img=False):
    """
    TODO@ZZK: Tensor & Batch
    :param layout_map:
    :param flag_img:
    :return:
    """
    layout = cv2.resize(layout_map.cpu().numpy(), (10, 10))
    # layout_map = F.interpolate(layout_map, size=(10, 10)).numpy()
    thres = sorted(layout.flatten(), reverse=True)[20]  ##取top20
    layout = (layout > thres) * 1
    if flag_img:
        layout *= 255
        layout = layout.astype('uint8')
    return layout


def norm_label(label, flag_img=False):
    """
    TODO@ZZK: Tensor & Batch
    :param label:
    :param flag_img:
    :return:
    """
    label = cv2.resize(label.cpu().numpy(), (10, 10))
    label = (label > 0.5) * 1
    assert np.sum(label) == 20, "Label Sum != 20"
    if flag_img:
        label *= 255
        label = label.astype('uint8')
    return label