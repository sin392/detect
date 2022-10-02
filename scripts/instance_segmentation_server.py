#!/usr/bin/env python3
from os.path import expanduser
from pathlib import Path

import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (InstanceSegmentationAction, InstanceSegmentationGoal,
                        InstanceSegmentationResult)
from detectron2.config import get_cfg

from entities.predictor import Predictor
from modules.ros.msg import Instance
from modules.ros.publisher import ImageMatPublisher


class Myserver:
    def __init__(self, cfg, instances_topic, image_topic, seg_topic):
        self.bridge = CvBridge()
        self.instances_topic = instances_topic
        self.image_topic = image_topic
        self.seg_topic = seg_topic

        self.predictor = Predictor(cfg)
        self.seg_publisher = ImageMatPublisher(seg_topic, queue_size=10)

        self.server = SimpleActionServer('instance_segmentation',
                                         InstanceSegmentationAction, self.listener_callback, False)
        self.server.start()

    def listener_callback(self, goal: InstanceSegmentationGoal):
        try:
            img_msg = goal.image
            img = self.bridge.imgmsg_to_cv2(img_msg)
            res = self.predictor.predict(img)
            frame_id = img_msg.header.frame_id
            stamp = img_msg.header.stamp

            seg = res.draw_instances(img[:, :, ::-1])
            self.seg_publisher.publish(seg, frame_id, stamp)

            instances = [Instance.from_instances(
                res, i) for i in range(res.num_instances)]
            result = InstanceSegmentationResult(instances)
            self.server.set_succeeded(result)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    rospy.init_node("instance_segmentaion_server", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    image_topic = rospy.get_param("image_topic")
    instances_topic = rospy.get_param("instances_topic")
    seg_topic = rospy.get_param("seg_topic")

    config_path = rospy.get_param("config", str(p.joinpath("resources/configs/config.yaml")))
    weight_path = rospy.get_param("weight", str(
        p.joinpath("outputs/2022_08_04_07_40/model_final.pth")))
    device = rospy.get_param("device", "cuda:0")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device

    Myserver(cfg, image_topic, instances_topic, seg_topic)

    rospy.spin()
