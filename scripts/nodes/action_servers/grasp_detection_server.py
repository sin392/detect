#!/usr/bin/env python3
from random import randint
from typing import List

import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, DetectedObject,
                        GraspDetectionAction, GraspDetectionGoal,
                        GraspDetectionResult, PointTuple2D)
from geometry_msgs.msg import Point, Pose
from modules.grasp import GraspDetector
from modules.ros.action_clients import (ComputeDepthThresholdClient,
                                        InstanceSegmentationClient, TFClient,
                                        VisualizeClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.publishers import DetectedObjectsPublisher
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, objects_topic: str, info_topic: str, enable_depth_filter: bool, enable_candidate_filter: bool):
        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.enable_candidate_filter = enable_candidate_filter
        cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        frame_size = (cam_info.height, cam_info.width)

        # Publishers
        self.objects_publisher = DetectedObjectsPublisher(objects_topic, queue_size=10)
        # Action Clients
        self.is_client = InstanceSegmentationClient()
        self.cdt_client = ComputeDepthThresholdClient() if enable_depth_filter else None
        self.tf_client = TFClient("base_link")
        self.visualize_client = VisualizeClient()
        # Others
        self.bridge = CvBridge()
        self.projector = PointProjector(cam_info)
        self.pose_estimator = PoseEstimator()
        self.grasp_detector = GraspDetector(finger_num=finger_num, frame_size=frame_size, unit_angle=15, margin=3)

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
            # TODO: compute n by camera distance
            opt_depth_th = None
            if self.cdt_client:
                opt_depth_th = self.cdt_client.compute(depth_msg, n=30)
                rospy.loginfo(opt_depth_th)
            objects: List[DetectedObject] = []
            for instance_msg in instances:
                mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)  # binary mask
                # ignore other than instances are located on top of stacks
                if opt_depth_th is not None:
                    masked_depth = depth * mask
                    if np.min(masked_depth[masked_depth > 0]) > opt_depth_th:
                        continue

                center = np.array(instance_msg.center)
                bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)

                bbox_short_side_px, bbox_long_side_px = bbox_handler.get_sides_2d()
                candidates = self.grasp_detector.detect(center, bbox_short_side_px, contour, depth, filter=self.enable_candidate_filter)
                if len(candidates) == 0:
                    continue

                # select best candidate
                # max_hand_width_px = self.projector.get_length_between_3d_points(p1_3d_c, p2_3d_c)
                target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
                best_cand = candidates[target_index]
                self.visualize_client.push_item(
                    Candidates(
                        candidates=[Candidate([PointTuple2D(pt) for pt in cnd.get_edges_on_rgb()]) for cnd in candidates],
                        bbox=bbox_handler.msg,
                        center=PointTuple2D(center),
                        target_index=target_index
                    )
                )

                # 3d projection
                points_c = [self.projector.screen_to_camera(uv, d) for uv, d in best_cand.get_edges_on_rgbd()]
                c_3d_c_on_surface = self.projector.screen_to_camera(center, depth[center[1]][center[0]])
                length_to_center = max([pt.z for pt in points_c]) - c_3d_c_on_surface.z
                c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z + length_to_center)

                points_and_center_w = self.tf_client.transform_points(header, (*points_c, c_3d_c))
                points_w = points_and_center_w[:-1]
                c_3d_w = points_and_center_w[-1]

                c_orientation = self.pose_estimator.get_orientation(depth, mask)
                bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector, depth)

                # TODO: DetectedObjectの３ポイント化
                objects.append(DetectedObject(
                    points=[pt.point for pt in points_w],
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
    finger_num = rospy.get_param("finger_num")
    objects_topic = rospy.get_param("objects_topic")
    info_topic = rospy.get_param("image_info_topic")
    enable_depth_filter = rospy.get_param("enable_depth_filter")
    enable_candidate_filter = rospy.get_param("enable_candidate_filter")

    GraspDetectionServer(
        "grasp_detection_server",
        finger_num=finger_num,
        objects_topic=objects_topic,
        info_topic=info_topic,
        enable_depth_filter=enable_depth_filter,
        enable_candidate_filter=enable_candidate_filter
    )
    rospy.spin()
