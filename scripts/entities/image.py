from typing import Union

import cv2
import numpy as np
from torch import Tensor


class BinaryMask:
    """
    mask: 1 or 255, (n, h, w)
    contour: used to calculate other values
    """

    def __init__(self, mask: Union[np.ndarray, Tensor]):
        self.mask: np.ndarray = np.asarray(mask, dtype=np.uint8)
        self.contour = self._get_contour()

    # private
    def _get_contour(self):
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
        else:
            contour = contours[0]

        return contour

    def get_center(self):
        mu = cv2.moments(self.contour)
        center = np.array([int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])])
        return np.int0(center)

    def get_rotated_bbox(self):
        # (upper_left, upper_right, lower_right, lower_left)
        return np.int0(cv2.boxPoints(cv2.minAreaRect(self.contour)))

    def get_area(self):
        return cv2.contourArea(self.contour)
