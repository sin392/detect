#!/usr/bin/env python3
from os.path import expanduser
from pathlib import Path
from typing import Union

import rospy
from cv_bridge import CvBridge
from detect.msg import Instance as RawInstance
from detect.msg import InstancesStamped, RotatedBoundingBox
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from entities.predictor import Instances as InstancesSchema
from entities.predictor import Predictor


class Instance(RawInstance):
    bridge = CvBridge()

    @classmethod
    def from_instances(cls, instances: InstancesSchema, index: int):
        return Instance(
            label=str(instances.labels[index]),
            score=instances.scores[index],
            bbox=RotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=cls.bridge.cv2_to_imgmsg(instances.mask_array[index])
        )


def callback(msg: Image, callback_args: Union[list, tuple]):
    instances_publisher: rospy.Publisher = callback_args[0]
    seg_publisher: rospy.Publisher = callback_args[1]
    bridge = CvBridge()

    try:
        rospy.loginfo(msg.header)
        img = bridge.imgmsg_to_cv2(msg)
        v = Visualizer(
            img[:, :, ::-1],
            metadata={},
            scale=0.5,
            # remove the colors of unsegmented pixels.
            # This option is only available for segmentation models
            instance_mode=ColorMode.IMAGE_BW
        )
        res = predictor.predict(img)

        # publish instances
        header = Header(stamp=msg.header.stamp,
                        frame_id=msg.header.frame_id)
        instances = [Instance.from_instances(
            res, i) for i in range(res.num_instances)]
        instances_publisher.publish(
            InstancesStamped(
                header=header,
                num_instances=res.num_instances,
                instances=instances,
            ))

        # publish image
        res_img = v.draw_instance_predictions(
            res._instances).get_image()[:, :, ::-1]
        res_img_msg = bridge.cv2_to_imgmsg(res_img, "rgb8")
        res_img_msg.header = header
        seg_publisher.publish(res_img_msg)

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
        "image_topic", "/body_camera/color/image_raw")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    predictor = Predictor(cfg)

    for image_topic in image_topics.split():
        instances_publisher = rospy.Publisher(
            image_topic + "/instances", InstancesStamped, queue_size=10)
        seg_publisher = rospy.Publisher(
            image_topic + "/seg", Image, queue_size=10)
        rospy.loginfo(f"sub: {image_topic}")
        rospy.Subscriber(image_topic, Image, callback=callback,
                         callback_args=(
                             instances_publisher,
                             seg_publisher
                         ))

    rospy.spin()
