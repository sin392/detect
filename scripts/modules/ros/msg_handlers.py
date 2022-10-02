#!/usr/bin/env python3
from typing import Tuple

import numpy as np
from cv_bridge import CvBridge
from detect.msg import Candidate, Instance, RotatedBoundingBox
from entities.predictor import PredictResult
from modules.ros.utils import numpy2multiarray
from std_msgs.msg import Int32MultiArray


class InstanceHandler:
    bridge = CvBridge()

    @classmethod
    def from_predict_result(cls, instances: PredictResult, index: int) -> Instance:
        return Instance(
            label=str(instances.labels[index]),
            score=instances.scores[index],
            bbox=RotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=cls.bridge.cv2_to_imgmsg(instances.mask_array[index]),
            contour=numpy2multiarray(
                Int32MultiArray, instances.contours[index])
        )


class RotatedBoundingBoxHandler:
    def __init__(self, msg: RotatedBoundingBox):
        self.msg = msg

    def tolist(self) -> Tuple[int, int, int, int]:
        return np.int0([self.msg.upper_left, self.msg.upper_right, self.msg.lower_right, self.msg.lower_left])

    def get_radiuses_on_image_plane(self) -> Tuple[float, float]:
        # ピクセルで表現される平面上の２点間の距離を算出する
        upper_left = np.array(self.msg.upper_left)
        upper_right = np.array(self.msg.upper_right)
        lower_left = np.array(self.msg.lower_left)

        radius_h = np.linalg.norm((upper_left - upper_right)) / 2
        radius_v = np.linalg.norm((upper_left - lower_left)) / 2
        short_radius = min(radius_h, radius_v)
        long_radius = max(radius_h, radius_v)

        return short_radius, long_radius


class CandidateHandler:
    def __init__(self, msg: Candidate):
        self.msg = msg

    def tolist(self) -> Tuple[int, int]:
        p1 = (self.msg.p1_u, self.msg.p1_v)
        p2 = (self.msg.p2_u, self.msg.p2_v)
        return (p1, p2)