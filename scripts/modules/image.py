from colorsys import hsv_to_rgb

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def gen_color_palette(n):
    hsv_array = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    rgb_array = np.array(list(
        map(lambda x: [int(v * 255) for v in hsv_to_rgb(*x)], hsv_array)), dtype=np.uint8)
    return rgb_array


def get_optimal_hist_th(depth: np.ndarray, n: int):
    valid_depth_values = depth[depth > 0]
    min_v, max_v = np.min(valid_depth_values), np.max(valid_depth_values)
    hist = cv2.calcHist(depth, channels=[0], mask=None, histSize=[
                        65536], ranges=[0, 65535])
    h_list = np.array([])
    for i in range(min_v, max_v + 1):
        t1 = np.sum(hist[i - n:i + n + 1])
        t2 = np.sum(hist[i - n * 2:i - n])
        t3 = np.sum(hist[i + n + 1:i + n * 2 + 1])
        res = t1 - t2 - t3
        h_list = np.append(h_list, res)
    sorted_h = np.argsort(h_list) + min_v
    return sorted_h


def extract_top_layer(img: np.ndarray, depth: np.ndarray, n: int):
    th = get_optimal_hist_th(depth, n)[0]
    flont_depth = np.where(depth < th, depth, 0)
    flont_depth_3d = np.array([flont_depth] * 3).transpose(1, 2, 0)
    flont_img = img * (flont_depth_3d / 65535)
    flont_img = cv2.normalize(
        flont_img, dst=None, alpha=0,
        beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    return flont_img, flont_depth


def extract_depth_between_two_points(depth, p1, p2):
    n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
    h, w = np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)
    res = map_coordinates(depth, np.vstack((h, w)))
    return res
