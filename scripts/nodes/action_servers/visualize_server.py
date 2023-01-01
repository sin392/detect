#!/usr/bin/env python3
from typing import List

import cv2
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, VisualizeCandidatesAction,
                        VisualizeCandidatesGoal)
# from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.publishers import ImageMatPublisher
from modules.visualize import convert_rgb_to_3dgray  # , draw_bbox
from modules.visualize import draw_candidate


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
        candidates_list: List[Candidates] = goal.candidates_list
        obj_target_index: int = goal.target_index
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp

        cnds_img = convert_rgb_to_3dgray(img)
        for obj_index, cnds_msg in enumerate(candidates_list):
            # draw bbox
            # bbox_handler = RotatedBoundingBoxHandler(cnds_msg.bbox)
            # cnds_img = draw_bbox(cnds_img, bbox_handler.tolist())
            # draw candidates
            candidates: List[Candidate] = cnds_msg.candidates
            obj_center_uv = cnds_msg.center.uv
            cnd_target_index = cnds_msg.target_index
            for cnd_index, cnd_msg in enumerate(candidates):
                cnd_center_uv = cnd_msg.center.uv
                coef = (1 - cnd_msg.score)
                color = (255, 255 * coef, 255 * coef)
                is_target = cnd_index == cnd_target_index
                for pt_msg in cnd_msg.insertion_points:
                    cnds_img = draw_candidate(cnds_img, cnd_center_uv, pt_msg.uv, color, is_target=is_target)
                cv2.circle(cnds_img, cnd_center_uv, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if obj_index == obj_target_index:
                best_score = candidates[cnd_target_index].score
                cv2.circle(cnds_img, cnd_center_uv, 6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                cv2.putText(cnds_img, f"{best_score:.2f}", (obj_center_uv[0] + 10, obj_center_uv[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(cnds_img, obj_center_uv, 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        self.publisher.publish(cnds_img, frame_id, stamp)


if __name__ == "__main__":
    pub_topic = rospy.get_param("candidates_img_topic")

    VisualizeServer("visualize_server", pub_topic)

    rospy.spin()
