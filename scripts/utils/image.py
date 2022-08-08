from colorsys import hsv_to_rgb
from typing import List, Tuple, Union

import cv2
import numpy as np


def gen_color_palette(n):
    hsv_array = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb_array = np.array(list(map(lambda x: [int(
        v*255) for v in hsv_to_rgb(*x)], hsv_array)), dtype=np.uint8)
    return rgb_array


class IndexedMask(np.ndarray):
    def __new__(cls, masks: Union[np.ndarray, List, Tuple]):
        masks = np.asarray(masks, dtype=np.uint8)
        assert masks.ndim == 3
        n, h, w = masks.shape
        # cast ndarray to IndexedMask
        self = np.zeros((h, w)).view(cls)
        # 要検証： masksが深度降順であることを前提にしている
        for i, mask in enumerate(masks):
            self[mask != 0] = i + 1

        self.n = n
        self.palette = gen_color_palette(n)
        return self

    def to_rgb(self):
        img = np.zeros((*self.shape, 3), dtype=np.uint8)
        for i in range(self.n):
            img[self == i+1] = self.palette[i]
        return img


def get_optimal_hist_th(img, n):
    min_v, max_v = np.min(img[img > 0]), np.max(img)
    hist = cv2.calcHist(img, channels=[0], mask=None,
                        histSize=[65536], ranges=[0, 65536])
    h_list = np.array([])
    for i in range(min_v, max_v+1):
        t1 = np.sum(hist[i-n:i+n+1])
        t2 = np.sum(hist[i-n*2:i-n])
        t3 = np.sum(hist[i+n+1:i+n*2+1])
        res = t1 - t2 - t3
        h_list = np.append(h_list, res)
    sorted_h = np.argsort(h_list) + min_v
    return sorted_h


def extract_top_layer(img, depth, n):
    th = get_optimal_hist_th(depth, n)[0]
    flont_depth = np.where(depth < th, depth, 0)
    flont_depth_3d = np.array([flont_depth]*3).transpose(1, 2, 0)
    flont_img = img * (flont_depth_3d / 65535)
    flont_img = cv2.normalize(
        flont_img, dst=None, alpha=0,
        beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    return flont_img, flont_depth
