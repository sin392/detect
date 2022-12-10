from colorsys import hsv_to_rgb

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from modules.const import UINT16MAX


def gen_color_palette(n):
    hsv_array = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    rgb_array = np.array(list(
        map(lambda x: [int(v * 255) for v in hsv_to_rgb(*x)], hsv_array)), dtype=np.uint8)
    return rgb_array


def transform_ddi(depth, n):
    mask = np.ones((n, n)).astype('uint8')  # erodeで使用するmaskはuint8
    # mask[n//2, n//2] = 0
    mask[1:-1, 1:-1] = 0  # 外周部以外は０に
    depth_min = cv2.erode(depth, mask, iterations=1)  # 最小値フィルタリング
    ddi = np.abs(depth.astype('int32') -
                 depth_min.astype('int32')).astype('uint16')
    return ddi


def compute_optimal_depth_thresh(depth, whole_mask, n):
    # ddiヒストグラムからddiしきい値を算出（物体のエッジに相当）
    ddi = transform_ddi(depth, n)
    hist_without_mask = cv2.calcHist([ddi], channels=[0], mask=None, histSize=[
                                     UINT16MAX], ranges=[0, UINT16MAX - 1])
    depth_values_on_mask = depth[whole_mask > 0]
    ddi_values_on_mask = ddi[whole_mask > 0]
    min_ddi, max_ddi = ddi_values_on_mask.min(), ddi_values_on_mask.max()

    h_list = []
    for i in range(min_ddi, max_ddi + 1):
        t1 = np.sum(hist_without_mask[i - n:i + n + 1])
        t2 = np.sum(hist_without_mask[i - n * 2:i - n])
        t3 = np.sum(hist_without_mask[i + n + 1:i + n * 2 + 1])
        res = t1 - t2 - t3
        h_list.append(res)
    sorted_h = np.argsort(h_list)  # argsortはデフォルト昇順
    optimal_ddi_thresh = sorted_h[-1] + min_ddi
    # ddiしきい値をdepthしきい値に変換
    optimal_depth_thresh = np.max(
        depth_values_on_mask[ddi_values_on_mask <= optimal_ddi_thresh])
    # optimal_depth_thresh = np.max(depth[ddi >= optimal_ddi_thresh])
    rounded_optimal_depth_thresh = np.int0(np.round(optimal_depth_thresh))

    return rounded_optimal_depth_thresh


def extract_flont_mask_with_thresh(depth, thresh, n):
    # flont_mask = np.where(depth <= thresh, whole_mask, 0).astype("uint8")
    flont_mask = np.where(depth <= thresh, 255, 0).astype("uint8")
    # 欠損ピクセルの補完
    closing_flont_mask = cv2.morphologyEx(
        flont_mask, cv2.MORPH_CLOSE, np.ones((n, n), np.uint8))
    # 膨張によりはみ出したピクセルの除去
    # final_flont_mask = np.where(whole_mask > 0, closing_flont_mask, 0)
    final_flont_mask = closing_flont_mask

    return final_flont_mask


def extract_flont_img(img, depth, whole_mask, n):
    optimal_depth_thresh = compute_optimal_depth_thresh(depth, whole_mask, n)
    flont_mask = extract_flont_mask_with_thresh(depth, optimal_depth_thresh, n)
    result_img = cv2.bitwise_and(img, img, mask=flont_mask)

    return result_img


def extract_depth_between_two_points(depth, p1, p2, mode="nearest", order=1):
    n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
    h, w = np.linspace(p1[1], p2[1], n), np.linspace(p1[0], p2[0], n)
    res = map_coordinates(depth, np.vstack((h, w)), mode=mode, order=order)
    return res
