import colorsys

import cv2
import numpy as np


def gen_color_palette(N, start_from_bg=False):
    if start_from_bg:
        HSV_array = [(0., 0., 0.)] + [(x*1.0/N, 0.5, 0.5) for x in range(N-1)]
    else:
        HSV_array = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_array = np.array(list(map(lambda x: [int(
        v*255) for v in colorsys.hsv_to_rgb(*x)], HSV_array)), dtype=np.uint8)
    return RGB_array


# convert indexed img to rgb img
def convert_to_rgb(index_colored_numpy, palette=None, n_colors=None):
    assert index_colored_numpy.dtype == np.uint8
    assert index_colored_numpy.ndim == 2
    if palette is None:
        palette = gen_color_palette(
            len(np.unique(index_colored_numpy)), start_from_bg=True)
    else:
        assert palette.shape[1] == 3 and palette.dtype == np.uint8 and palette.ndim == 2
    if n_colors is None:
        n_colors = palette.shape[0]
    reduced = index_colored_numpy.copy()
    reduced[index_colored_numpy > n_colors] = 0  # 不要なクラスを0とする
    expanded_img = np.eye(n_colors, dtype=np.int32)[
        reduced]  # [H, W, n_colors] int32
    use_pallete = palette[:n_colors].astype(np.int32)  # [n_colors, 3] int32
    return np.dot(expanded_img, use_pallete).astype(np.uint8)


def draw_bbox(img, box, color=(0, 255, 0), copy=False):
    res_img = img.copy() if copy else img
    cv2.drawContours(res_img, [np.int0(box)], 0, color, 2)
    return res_img


def draw_candidates(img, candidates, color=(0, 0, 255), targer_color=(255, 0, 0), target_index=None, copy=False):
    res_img = img.copy() if copy else img
    for i, (p1, p2) in enumerate(candidates):
        cv2.line(res_img, p1, p2, targer_color if i == target_index else color)
    return res_img


def draw_candidates_and_boxes(img, candidates_list, boxes, target_indexes=None, gray=False, copy=False):
    res_img = img.copy() if copy else img
    if gray:
        res_img = convert_rgb_to_3dgray(img)
    if not target_indexes:
        target_indexes = [None] * len(candidates_list)
    for candidates, box, target_index in zip(candidates_list, boxes, target_indexes):
        draw_bbox(res_img, box, copy=False)
        draw_candidates(res_img, candidates,
                        target_index=target_index, copy=False)
    return res_img


def convert_1dgray_to_3dgray(gray):
    gray_1d = gray[:, :, np.newaxis] if len(gray.shape) == 2 else gray
    gray_3d = cv2.merge([gray_1d] * 3)
    return gray_3d


def convert_rgb_to_3dgray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = convert_1dgray_to_3dgray(gray)
    return gray_3d
