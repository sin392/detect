#!/usr/bin/env python3
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import VisualizeCandidatesAction, VisualizeCandidatesGoal

from modules.ros.msg import Candidate, RotatedBoundingBox
from modules.ros.publisher import ImageMatPublisher
from modules.visualize import convert_rgb_to_3dgray, draw_bbox, draw_candidate


class VisualizeServer:
    def __init__(self, name: str, pub_topic: str):
        rospy.init_node(name)

        self.bridge = CvBridge()
        self.publisher = ImageMatPublisher(pub_topic, queue_size=10)
        self.server = SimpleActionServer(name, VisualizeCandidatesAction, self.callback, False)
        self.server.start()

    def callback(self, goal: VisualizeCandidatesGoal):
        img_msg = goal.base_image
        img = self.bridge.imgmsg_to_cv2(img_msg)
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp

        cnds_img = convert_rgb_to_3dgray(img)
        for cnds_msg in goal.candidates_list:
            bbox = RotatedBoundingBox.tolist(cnds_msg.bbox)
            cnds_img = draw_bbox(cnds_img, bbox)
            target_index = cnds_msg.target_index
            for i, cnd_msg in enumerate(cnds_msg.candidates):
                p1, p2 = Candidate.tolist(cnd_msg)
                is_target = i == target_index
                cnds_img = draw_candidate(cnds_img, p1, p2, is_target=is_target)

        self.publisher.publish(cnds_img, frame_id, stamp)


if __name__ == "__main__":
    pub_topic = rospy.get_param("candidates_topic")

    VisualizeServer("visualize_server", pub_topic)

    rospy.spin()
