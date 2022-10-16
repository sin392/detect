#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (DepthFilterAction, DepthFilterActionGoal,
                        DepthFilterResult)


class DepthFilterServer:
    def __init__(self, name: str, n: int = 30):
        rospy.init_node(name, log_level=rospy.INFO)

        self.n = n

        self.bridge = CvBridge()
        self.server = SimpleActionServer(name, DepthFilterAction, self.callback, False)
        self.server.start()

    def callback(self, goal: DepthFilterActionGoal):
        img_msg = goal.rgb
        depth_msg = goal.depth

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)

            optimal_th = self.get_optimal_hist_th(depth, self.n)[0]
            rospy.loginfo(optimal_th)
            mask_1c = np.where(depth < optimal_th, 1, 0)[:, :, np.newaxis]

            filtered_img = (img * mask_1c).astype("uint8")
            filtered_img_msg = self.bridge.cv2_to_imgmsg(filtered_img, "rgb8")
            filtered_img_msg.header = img_msg.header

            self.server.set_succeeded(DepthFilterResult(filtered_img_msg))

        except Exception as err:
            rospy.logerr(err)

    def get_optimal_hist_th(self, depth: np.ndarray, n: int):
        valid_depth_values = depth[depth > 0]
        min_v, max_v = np.min(valid_depth_values), np.max(valid_depth_values)
        hist = cv2.calcHist(depth, channels=[0], mask=None, histSize=[65536], ranges=[0, 65535])
        h_list = np.array([])
        for i in range(min_v, max_v + 1):
            t1 = np.sum(hist[i - n:i + n + 1])
            t2 = np.sum(hist[i - n * 2:i - n])
            t3 = np.sum(hist[i + n + 1:i + n * 2 + 1])
            res = t1 - t2 - t3
            h_list = np.append(h_list, res)
        sorted_h = np.argsort(h_list) + min_v
        return sorted_h


if __name__ == "__main__":
    # TODO: nはリクエストでわたすべきでは
    DepthFilterServer("depth_filter_server", n=30)

    rospy.spin()
