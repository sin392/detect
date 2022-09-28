#!/usr/bin/env python3
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import VisualizeCandidatesAction

from modules.ros.publisher import ImageMatPublisher
from modules.ros.utils import bboxmsg2list
from modules.visualize import convert_rgb_to_3dgray, draw_bbox, draw_candidate


class MyServer:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_topic = rospy.get_param("candidates_topic")
        self.publisher = ImageMatPublisher(self.pub_topic, queue_size=10)
        self.server = SimpleActionServer('visualize',
                                         VisualizeCandidatesAction, self.listener_callback, False)
        self.server.start()

    def listener_callback(self, goal):
        img_msg = goal.base_image
        img = self.bridge.imgmsg_to_cv2(img_msg)
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp

        cnds_img = convert_rgb_to_3dgray(img)
        for cnds_msg in goal.candidates_list:
            bbox = bboxmsg2list(cnds_msg.bbox)
            cnds_img = draw_bbox(cnds_img, bbox)
            for cnd_msg in cnds_msg.candidates:
                p1 = (cnd_msg.p1_u, cnd_msg.p1_v)
                p2 = (cnd_msg.p2_u, cnd_msg.p2_v)
                cnds_img = draw_candidate(cnds_img, p1, p2, is_target=False)

        self.publisher.publish(cnds_img, frame_id, stamp)


if __name__ == "__main__":
    rospy.init_node("visualize_server")
    MyServer()

    rospy.spin()
