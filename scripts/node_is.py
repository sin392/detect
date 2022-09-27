#!/usr/bin/env python3
from os.path import expanduser
from pathlib import Path
from typing import Union

import rospy
from cv_bridge import CvBridge
from detect.msg import Instance as RawInstance
from detect.msg import RotatedBoundingBox
from detectron2.config import get_cfg
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from entities.predictor import Predictor, PredictResult
from modules.ros.publisher import ImageMatPublisher, InstancesPublisher
from modules.ros.utils import numpy2multiarray

bridge = CvBridge()


class Instance(RawInstance):
    global bridge

    @classmethod
    def from_instances(cls, instances: PredictResult, index: int):
        return Instance(
            label=str(instances.labels[index]),
            score=instances.scores[index],
            bbox=RotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=bridge.cv2_to_imgmsg(instances.mask_array[index]),
            contour=numpy2multiarray(
                Int32MultiArray, instances.contours[index])
        )


def callback(msg: Image, callback_args: Union[list, tuple]):
    predictor: Predictor = callback_args[0]
    instances_publisher: InstancesPublisher = callback_args[1]
    seg_publisher: ImageMatPublisher = callback_args[2]

    try:
        rospy.loginfo(msg.header)
        img = bridge.imgmsg_to_cv2(msg)
        res = predictor.predict(img)
        frame_id = msg.header.frame_id
        stamp = msg.header.stamp

        # publish instances
        instances = [Instance.from_instances(
            res, i) for i in range(res.num_instances)]
        instances_publisher.publish(
            res.num_instances, instances, frame_id, stamp)

        # publish image
        res_img = res.draw_instances(img[:, :, ::-1])
        seg_publisher.publish(res_img, frame_id, stamp)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    rospy.init_node("instance_segmentaion_node", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    config_path = rospy.get_param(
        "config", str(p.joinpath("resources/configs/config.yaml")))
    weight_path = rospy.get_param("weight", str(
        p.joinpath("outputs/2022_08_04_07_40/model_final.pth")))
    device = rospy.get_param("device", "cuda:0")
    image_topics = rospy.get_param(
        "image_topic", "/myrobot/body_camera/color/image_raw")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    predictor = Predictor(cfg)

    for image_topic in image_topics.split():
        instances_publisher = InstancesPublisher(
            image_topic + "/instances", queue_size=10)
        seg_publisher = ImageMatPublisher(
            image_topic + "/seg", queue_size=10)
        rospy.loginfo(f"sub: {image_topic}")
        rospy.Subscriber(image_topic, Image, callback=callback,
                         callback_args=(
                             predictor,
                             instances_publisher,
                             seg_publisher
                         ))

    rospy.spin()
