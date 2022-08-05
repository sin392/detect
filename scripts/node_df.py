#!/usr/bin/env python3
from os.path import expanduser
from pathlib import Path
from typing import Union

import cv2
import message_filters as mf
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from utils.image import get_optimal_hist_th


def callback(img_msg: Image, depth_msg: Image, *callback_args: Union[list, tuple]):
    publisher: rospy.Publisher = callback_args[0]
    bridge = CvBridge()

    try:
        rospy.loginfo(img_msg.header)
        rospy.loginfo(depth_msg.header)
        img = bridge.imgmsg_to_cv2(img_msg)
        depth = bridge.imgmsg_to_cv2(depth_msg)

        optimal_th = get_optimal_hist_th(depth, n=30)[0]
        rospy.loginfo(optimal_th)
        mask_1c = np.where(depth < 1500, 1, 0)[:, :, np.newaxis]

        # rospy.loginfo(
        #     f"{img.shape}, {mask_1c.shape} {cv2.merge((mask_1c, mask_1c, mask_1c)).shape}")
        # res_img = cv2.bitwise_and(img, cv2.merge((mask_1c, mask_1c, mask_1c)))
        res_img = (img * mask_1c).astype("uint8")
        res_img_msg = bridge.cv2_to_imgmsg(res_img, "rgb8")

        # outputs info publish
        header = Header(stamp=rospy.get_rostime(),
                        frame_id=img_msg.header.frame_id)
        res_img_msg.header = header
        publisher.publish(res_img_msg)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    rospy.init_node("depth_filter_node", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    image_topics = rospy.get_param(
        "image_topic", "/body_camera/color/image_raw")

    fps = 10.
    delay = 1 / fps * 0.5
    # for image_topic in image_topics.split():
    image_topic = image_topics

    depth_topic = image_topic.replace("color", "aligned_depth_to_color")
    publisher = rospy.Publisher(
        image_topic + "/filtered", Image, queue_size=10)
    rospy.loginfo(f"sub: {image_topic}, {depth_topic}")
    img_subscriber = mf.Subscriber(image_topic, Image)
    depth_subscriber = mf.Subscriber(depth_topic, Image)
    subscribers = [img_subscriber, depth_subscriber]

    ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
    # ts = mf.TimeSynchronizer(subscribers, 10)
    ts.registerCallback(callback, (publisher))

    rospy.spin()
