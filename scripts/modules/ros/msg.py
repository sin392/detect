#!/usr/bin/env python3
from cv_bridge import CvBridge
from detect.msg import Instance as RawInstance
from detect.msg import RotatedBoundingBox
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
            bbox=RotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=cls.bridge.cv2_to_imgmsg(instances.mask_array[index]),
            contour=numpy2multiarray(
                Int32MultiArray, instances.contours[index])
        )
