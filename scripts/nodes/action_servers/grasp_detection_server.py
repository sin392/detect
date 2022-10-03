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
from geometry_msgs.msg import Point, Pose
from modules.grasp import ParallelGraspDetector
from modules.ros.action_clients import (InstanceSegmentationClient, TFClient,
                                        VisualizeClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.publishers import DetectedObjectsPublisher
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class GraspDetectionServer:
    def __init__(self, name: str, objects_topic: str, info_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        frame_size = (cam_info.height, cam_info.width)

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
        self.grasp_detector = ParallelGraspDetector(frame_size=frame_size, unit_angle=15, margin=3)

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
                center = np.array(instance_msg.center)
                bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)
                mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)

                bbox_short_side_px, bbox_long_side_px = bbox_handler.get_sides_2d()
                candidates = self.grasp_detector.detect(center, bbox_short_side_px, contour, depth, filter=True)
                if len(candidates) == 0:
                    continue

                # select best candidate
                # max_hand_width_px = self.projector.get_length_between_3d_points(p1_3d_c, p2_3d_c)
                target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
                best_cand = candidates[target_index]
                self.visualize_client.push_item(
                    Candidates(
                        [Candidate(cnd.p1_u, cnd.p1_v, cnd.p2_u, cnd.p2_v) for cnd in candidates],
                        bbox_handler.msg,
                        target_index
                    )
                )

                # 3d projection
                p1_3d_c, p2_3d_c = [self.projector.screen_to_camera(uv, d) for uv, d in best_cand.get_candidate_points_on_rgbd()]
                c_3d_c_on_surface = self.projector.screen_to_camera(*best_cand.get_center_on_rgbd())
                length_to_center = max(p1_3d_c.z, p2_3d_c.z) - c_3d_c_on_surface.z
                c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface + length_to_center)

                p1_3d_w, p2_3d_w, c_3d_w = self.tf_client.transform_points(header, (p1_3d_c, p2_3d_c, c_3d_c))

                c_orientation = self.pose_estimator.get_orientation(depth, mask)
                bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector)

                objects.append(DetectedObject(
                    p1=p1_3d_w.point,
                    p2=p2_3d_w.point,
                    center_pose=Pose(
                        position=c_3d_w.point,
                        orientation=c_orientation
                    ),
                    short_radius=bbox_short_side_3d / 2,
                    long_radius=bbox_long_side_3d / 2
                ))

            self.visualize_client.visualize_stacked_candidates(img_msg)
            self.server.set_succeeded(GraspDetectionResult(objects))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    objects_topic = rospy.get_param("objects_topic")
    info_topic = rospy.get_param("image_info_topic")

    GraspDetectionServer(
        "grasp_detection_server",
        objects_topic=objects_topic,
        info_topic=info_topic
    )
    rospy.spin()
