#!/usr/bin/env python3
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
    @classmethod
    def tolist(cls, msg: RawRotatedBoundingBox):
        return np.int0([msg.upper_left, msg.upper_right, msg.lower_right, msg.lower_left])


class Candidate(RawCandidate):
    @classmethod
    def tolist(cls, msg: RawCandidate):
        p1 = (msg.p1_u, msg.p1_v)
        p2 = (msg.p2_u, msg.p2_v)
        return (p1, p2)
