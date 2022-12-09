#!/usr/bin/env python3
from random import randint
from typing import List

import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, DetectedObject,
                        GraspDetectionAction, GraspDetectionDebugInfo,
                        GraspDetectionGoal, GraspDetectionResult, PointTuple2D)
from geometry_msgs.msg import Point, Pose, PoseStamped
from modules.grasp import GraspDetector
from modules.ros.action_clients import (ComputeDepthThresholdClient,
                                        InstanceSegmentationClient, TFClient,
                                        VisualizeClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, finger_width_mm: int, hand_mount_rotation: int, info_topic: str, enable_depth_filter: bool, enable_candidate_filter: bool, debug: bool):
        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.base_angle = 360 // finger_num
        self.finger_width_mm = finger_width_mm  # length between center and edge
        self.hand_mount_rotation = hand_mount_rotation
        self.enable_candidate_filter = enable_candidate_filter
        self.debug = debug
        cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        frame_size = (cam_info.height, cam_info.width)

        # Publishers
        self.dbg_info_publisher = rospy.Publisher("/grasp_detection_server/result/debug", GraspDetectionDebugInfo, queue_size=10) if debug else None
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

        # convert unit
        # ref: https://qiita.com/srs/items/e3412e5b25477b46f3bd
        flatten_corrected_params = cam_info.P
        fp_x, fp_y = flatten_corrected_params[0], flatten_corrected_params[5]
        self.fp = (fp_x + fp_y) / 2

        self.server = SimpleActionServer(name, GraspDetectionAction, self.callback, False)
        self.server.start()

    def callback(self, goal: GraspDetectionGoal):
        print("receive request")
        img_msg = goal.image
        depth_msg = goal.depth
        # rgbとdepthで軸の向きが違った。想定しているカメラ座標系はrgbの方
        # frame_id = depth_msg.header.frame_id
        # stamp = depth_msg.header.stamp
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp
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
            candidates_list: List[Candidates] = []
            for instance_msg in instances:
                mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)  # binary mask
                # ignore other than instances are located on top of stacks
                if opt_depth_th is not None:
                    masked_depth = depth * mask
                    if np.min(masked_depth[masked_depth > 0]) > opt_depth_th:
                        continue

                center = np.array(instance_msg.center)
                center_d_mm = depth[center[1]][center[0]]
                bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)

                # bbox_short_side_px, bbox_long_side_px = bbox_handler.get_sides_2d()
                finger_width_px = self.finger_width_mm * self.fp * center_d_mm / 1000000
                # finger_width_mm = 150
                # finger_width_px = finger_width_mm * self.fp * (center_d_mm - 1000) / (1000 ** 2)

                candidate_radius = finger_width_px
                candidates = self.grasp_detector.detect(center, candidate_radius, contour, depth, filter=self.enable_candidate_filter)
                if len(candidates) == 0:
                    continue

                # select best candidate
                # max_hand_width_px = self.projector.get_length_between_3d_points(p1_3d_c, p2_3d_c)
                valid_candidates = [cnd for cnd in candidates if cnd.is_valid] if self.enable_candidate_filter else candidates
                target_index = randint(0, len(valid_candidates) - 1) if len(valid_candidates) != 0 else 0
                best_cand = valid_candidates[target_index]
                candidates_list.append(
                    Candidates(
                        candidates=[Candidate([PointTuple2D(pt) for pt in cnd.get_insertion_points_uv()]) for cnd in valid_candidates],
                        bbox=bbox_handler.msg,
                        center=PointTuple2D(center),
                        target_index=target_index
                    )
                )

                # 3d projection
                points_c = [self.projector.screen_to_camera(uv, d_mm) for uv, d_mm in best_cand.get_insertion_points_uvd()]
                c_3d_c_on_surface = self.projector.screen_to_camera(center, center_d_mm)
                length_to_center = max([pt.z for pt in points_c]) - c_3d_c_on_surface.z
                c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z + length_to_center)

                points_and_center_w = self.tf_client.transform_points(header, (*points_c, c_3d_c))
                points_w = points_and_center_w[:-1]
                c_3d_w = points_and_center_w[-1]

                c_orientation = self.pose_estimator.get_orientation(depth, mask)
                bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector, depth)

                # NOTE: unclockwise seen from image plane is positive in cnd.angle, so convert as rotate on z-axis
                angle = -best_cand.angle + self.hand_mount_rotation
                angles = []
                for i in range(1, self.finger_num + 1):
                    raw_rotated_angle = angle - (self.base_angle * i)
                    rotated_angle = raw_rotated_angle + 360 if raw_rotated_angle < -360 else raw_rotated_angle
                    reversed_rotated_angle = rotated_angle + 360
                    angles.extend([rotated_angle, reversed_rotated_angle])
                angles.sort(key=abs)
                objects.append(DetectedObject(
                    points=[pt.point for pt in points_w],
                    center_pose=PoseStamped(
                        Header(frame_id="base_link"),
                        Pose(
                            position=c_3d_w.point,
                            orientation=c_orientation
                        )
                    ),
                    angles=angles,
                    short_radius=bbox_short_side_3d / 2,
                    long_radius=bbox_long_side_3d / 2,
                    length_to_center=length_to_center
                ))

            self.visualize_client.visualize_candidates(img_msg, candidates_list)
            if self.dbg_info_publisher:
                self.dbg_info_publisher.publish(GraspDetectionDebugInfo(header, candidates_list))
            self.server.set_succeeded(GraspDetectionResult(header, objects))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    finger_num = rospy.get_param("finger_num")
    finger_width_mm = rospy.get_param("finger_width_mm")
    hand_mount_rotation = rospy.get_param("hand_mount_rotation")
    info_topic = rospy.get_param("image_info_topic")
    enable_depth_filter = rospy.get_param("enable_depth_filter")
    enable_candidate_filter = rospy.get_param("enable_candidate_filter")
    debug = rospy.get_param("debug")

    GraspDetectionServer(
        "grasp_detection_server",
        finger_num=finger_num,
        finger_width_mm=finger_width_mm,
        hand_mount_rotation=hand_mount_rotation,
        info_topic=info_topic,
        enable_depth_filter=enable_depth_filter,
        enable_candidate_filter=enable_candidate_filter,
        debug=debug
    )
    rospy.spin()
