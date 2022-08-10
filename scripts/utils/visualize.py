import colorsys

import cv2
import numpy as np


def draw_bbox(img, box, color=(0, 255, 0), **kwargs):
    cv2.drawContours(res_img, [np.int0(box)], 0, color, 2, **kwargs)
    return res_img


def draw_candidates(img, candidates, color=(0, 0, 255), targer_color=(255, 0, 0), target_index=None, **kwargs):
    for i, (p1, p2) in enumerate(candidates):
        cv2.line(img, p1, p2, targer_color if i ==
                 target_index else color, **kwargs)


def draw_candidates_and_boxes(img, candidates_list, boxes, target_indexes=None, gray=False, **kwargs):
    if gray:
        img = convert_rgb_to_3dgray(img)
    if not target_indexes:
        target_indexes = [None] * len(candidates_list)
    for candidates, box, target_index in zip(candidates_list, boxes, target_indexes):
        draw_bbox(img, box, **kwargs)
        draw_candidates(img, candidates,
                        target_index=target_index, **kwargs)


def convert_1dgray_to_3dgray(gray):
    gray_1d = gray[:, :, np.newaxis] if len(gray.shape) == 2 else gray
    gray_3d = cv2.merge([gray_1d] * 3)
    return gray_3d


def convert_rgb_to_3dgray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = convert_1dgray_to_3dgray(gray)
    return gray_3d
