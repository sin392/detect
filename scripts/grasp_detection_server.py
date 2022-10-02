#!/usr/bin/env python3
from random import randint
from typing import List

import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, DetectedObject,
                        GraspDetectionAction, GraspDetectionGoal,
                        GraspDetectionResult)
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from modules.const import FRAME_SIZE
from modules.grasp import ParallelGraspDetector
from modules.ros.action_client import (InstanceSegmentationClient, TFClient,
                                       VisualizeClient)
from modules.ros.msg import RotatedBoundingBox
from modules.ros.publisher import DetectedObjectsPublisher
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy


class GraspDetectionServer:
    def __init__(self, name: str, objects_topic: str, info_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        cam_info = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)

        # Publishers
        self.objects_publisher = DetectedObjectsPublisher(objects_topic, queue_size=10)
        # Action Clients
        self.is_client = InstanceSegmentationClient()
        self.tf_client = TFClient("base_link")
        self.visualize_client = VisualizeClient()
        # Others
        self.bridge = CvBridge()
        self.projector = PointProjector(cam_info)
        self.pose_estimator = PoseEstimator()
        self.grasp_detector = ParallelGraspDetector(frame_size=FRAME_SIZE, unit_angle=15, margin=3)

        self.server = SimpleActionServer(name, GraspDetectionAction, self.callback, False)
        self.server.start()

    def callback(self, goal: GraspDetectionGoal):
        img_msg = goal.image
        depth_msg = goal.depth
        frame_id = depth_msg.header.frame_id
        stamp = depth_msg.header.stamp
        header = Header(frame_id=frame_id, stamp=stamp)
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg)
            objects: List[DetectedObject] = []
            for instance_msg in instances:
                center = instance_msg.center
                bbox_msg = RotatedBoundingBox(instance_msg.bbox)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)
                mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)

                # NOTE: following radiuses are [pixel (float)]
                short_radius_2d, long_radius_2d = bbox_msg.get_radiuses_on_image_plane()
                candidates = self.grasp_detector.detect(center, short_radius_2d, contour, depth, filter=True)
                if len(candidates) == 0:
                    continue

                # select best candidate
                target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
                self.visualize_client.push_item(Candidates([Candidate(cnd.p1, cnd.p2) for cnd in candidates], bbox_msg, target_index))

                # 3d projection
                p1_3d_c, p2_3d_c = [self.projector.pixel_to_3d(*point[::-1], depth) for point in candidates[target_index]]
                # NOTE: following radiuses are [mm]
                long_radius_3d = np.linalg.norm(
                    np.array([p1_3d_c.x, p1_3d_c.y, p1_3d_c.z]) - np.array([p2_3d_c.x, p2_3d_c.y, p2_3d_c.z])
                ) / 2
                short_radius_3d = long_radius_3d / 2

                c_3d_c = self.projector.pixel_to_3d(
                    *center[::-1],
                    depth,
                    margin_mm=short_radius_3d  # 中心点は物体表面でなく中心座標を取得したいのでmargin_mmを指定
                )

                # transform from camera to world
                p1_3d_w, p2_3d_w, c_3d_w = self.tf_client.transform_points(header, (p1_3d_c, p2_3d_c, c_3d_c))

                c_orientation = self.pose_estimator.get_orientation(depth, mask)

                objects.append(DetectedObject(
                    p1=p1_3d_w.point,
                    p2=p2_3d_w.point,
                    center_pose=Pose(
                        position=c_3d_w.point,
                        orientation=c_orientation
                    ),
                    short_radius=short_radius_3d,
                    long_radius=long_radius_3d
                ))

            self.visualize_client.visualize_stacked_candidates(img_msg)
            self.server.set_succeeded(GraspDetectionResult(objects))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    objects_topic = rospy.get_param("objects_topic")
    info_topic = rospy.get_param("depth_info_topic")

    GraspDetectionServer(
        "grasp_detection_server",
        objects_topic=objects_topic,
        info_topic=info_topic
    )
    rospy.spin()
