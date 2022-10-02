#!/usr/bin/env python3
from typing import Tuple

import numpy as np
from cv_bridge import CvBridge
from detect.msg import Candidate as RawCandidate
from detect.msg import Instance as RawInstance
from detect.msg import RotatedBoundingBox as RawRotatedBoundingBox
from entities.predictor import PredictResult
from modules.ros.utils import numpy2multiarray
from std_msgs.msg import Int32MultiArray


class Instance(RawInstance):
    bridge = CvBridge()

    @classmethod
    def from_instances(cls, instances: PredictResult, index: int):
        return Instance(
            label=str(instances.labels[index]),
            score=instances.scores[index],
            bbox=RawRotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=cls.bridge.cv2_to_imgmsg(instances.mask_array[index]),
            contour=numpy2multiarray(
                Int32MultiArray, instances.contours[index])
        )


class RotatedBoundingBox(RawRotatedBoundingBox):
    def tolist(self) -> Tuple[int, int, int, int]:
        return np.int0([self.upper_left, self.upper_right, self.lower_right, self.lower_left])

    def get_radiuses_on_image_plane(self) -> Tuple[float, float]:
        # ピクセルで表現される平面上の２点間の距離を算出する
        radius_h = np.linalg.norm((self.upper_left[0] - self.upper_right[0]), (self.upper_left[1] - self.upper_right[1])) / 2
        radius_v = np.linalg.norm((self.upper_left[0] - self.lower_left[0]), (self.upper_left[1] - self.lower_left[1])) / 2
        short_radius = min(radius_h, radius_v)
        long_radius = max(radius_h, radius_v)

        return short_radius, long_radius


class Candidate(RawCandidate):
    def tolist(self) -> Tuple[int, int]:
        p1 = (self.p1_u, self.p1_v)
        p2 = (self.p2_u, self.p2_v)
        return (p1, p2)
