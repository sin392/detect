#!/usr/bin/env python3
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from detectron2.config import get_cfg
from inference import Predictor
from sensor_msgs.msg import Image

bridge = CvBridge()


def callback(msg):
    try:
        img = bridge.imgmsg_to_cv2(msg)
        rospy.loginfo(img)
    except Exception as err:
        rospy.logerr(err)


rospy.init_node("instance_segmentaion_node", log_level=rospy.DEBUG)
rospy.Subscriber("/body_camera/color/image_raw",
                 Image, callback=callback)
# r = rospy.Rate(10)

# pridictor = Predictor()

p = Path("~/workspace/catkin_ws")
dataset_path = p.joinpath("datasets")

cfg = get_cfg()
cfg.merge_from_file(str(p.joinpath("configs", "config.yaml")))
cfg.MODEL.DEVICE = args.device
cfg.MODEL.WEIGHTS = args.weight or f"{cfg.OUTPUT_DIR}/2022_08_04_07_40/model_final.pth"


rospy.spin()
