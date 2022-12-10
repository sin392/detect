#!/usr/bin/env python3
from time import time
from typing import List

import cv2
import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, DetectedObject,
                        GraspDetectionAction, GraspDetectionDebugInfo,
                        GraspDetectionGoal, GraspDetectionResult, PointTuple2D)
from geometry_msgs.msg import Point, Pose, PoseStamped
from modules.const import UINT16MAX
from modules.grasp import GraspDetector
from modules.image import extract_flont_mask_with_thresh
from modules.ros.action_clients import (ComputeDepthThresholdClient,
                                        InstanceSegmentationClient, TFClient,
                                        VisualizeClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, unit_angle: int, hand_radius_mm: int, finger_radius_mm: int, hand_mount_rotation: int,
                 elements_th: float, center_diff_th: float, el_insertion_th: float, el_contact_th: float, el_bw_depth_th: float,
                 info_topic: str, enable_depth_filter: bool, enable_candidate_filter: bool, debug: bool):
        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.unit_angle = unit_angle
        self.base_angle = 360 // finger_num
        self.hand_radius_mm = hand_radius_mm  # length between center and edge
        self.finger_radius_mm = finger_radius_mm
        self.hand_mount_rotation = hand_mount_rotation
        self.elements_th = elements_th
        self.center_diff_th = center_diff_th
        self.el_insertion_th = el_insertion_th
        self.el_contact_th = el_contact_th
        self.el_bw_depth_th = el_bw_depth_th
        self.enable_candidate_filter = enable_candidate_filter
        self.debug = debug
        cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        frame_size = (cam_info.height, cam_info.width)

        # convert unit
        # ref: https://qiita.com/srs/items/e3412e5b25477b46f3bd
        flatten_corrected_params = cam_info.P
        fp_x, fp_y = flatten_corrected_params[0], flatten_corrected_params[5]
        fp = (fp_x + fp_y) / 2

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
        self.grasp_detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                                            finger_radius_mm=finger_radius_mm,
                                            unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                                            elements_th=elements_th, center_diff_th=center_diff_th,
                                            el_insertion_th=el_insertion_th, el_contact_th=el_contact_th,
                                            el_bw_depth_th=el_bw_depth_th)

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
            start_time = time()
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg)
            # TODO: depthしきい値を求めるためにmerged_maskが必要だが非効率なので要改善
            masks = [self.bridge.imgmsg_to_cv2(instance_msg.mask) for instance_msg in instances]
            # TODO: compute n by camera distance
            if self.cdt_client:
                merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0).astype("uint8")
                min_d = depth[merged_mask > 0].min()
                opt_depth_th = self.cdt_client.compute(depth_msg, min_d=min_d, n=5)
                flont_mask = extract_flont_mask_with_thresh(depth, merged_mask, opt_depth_th, n=5)
                flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
                vis_base_img_msg = self.bridge.cv2_to_imgmsg(flont_img)

                rospy.loginfo(opt_depth_th)
            else:
                opt_depth_th = UINT16MAX
                vis_base_img_msg = img_msg
            objects: List[DetectedObject] = []  # 空のものは省く
            candidates_list: List[Candidates] = []  # 空のものも含む
            for instance_msg, mask in zip(instances, masks):
                # mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)  # binary mask
                # ignore other than instances are located on top of stacks
                # TODO: しきい値で切り出したマスク内に含まれないインスタンスはスキップ
                # TODO: スキップされてもcenterは描画したい
                instance_max_d = depth[mask > 0].max()
                if instance_max_d > opt_depth_th:
                    continue

                center = np.array(instance_msg.center)
                center_d_mm = depth[center[1], center[0]]
                bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)

                # bbox_short_side_px, bbox_long_side_px = bbox_handler.get_sides_2d()
                # detect candidates
                candidates = self.grasp_detector.detect(center=center, depth=depth, contour=contour)
                if len(candidates) == 0:
                    continue
                # select best candidate
                valid_candidates = [cnd for cnd in candidates if cnd.is_valid] if self.enable_candidate_filter else candidates
                is_valid = len(valid_candidates) > 0
                valid_scores = [cnd.total_score for cnd in valid_candidates]
                target_index = np.argmax(valid_scores) if is_valid else 0

                # candidates_list は空のものも含む
                candidates_list.append(
                    Candidates(
                        candidates=[
                            Candidate(
                                [PointTuple2D(pt) for pt in cnd.get_insertion_points_uv()],
                                [PointTuple2D(pt) for pt in cnd.get_contact_points_uv()],
                                cnd.total_score,
                                cnd.is_valid
                            )
                            for cnd in valid_candidates
                        ],
                        # bbox=bbox_handler.msg,
                        center=PointTuple2D(center),
                        target_index=target_index
                    )
                )

                # TODO: is_frameinの判定冗長なので要整理
                include_any_frameout = not np.any([cnd.is_framein for cnd in valid_candidates])
                if include_any_frameout or not is_valid:
                    continue

                best_cand = valid_candidates[target_index]
                # 3d projection
                insertion_points_c = [self.projector.screen_to_camera(uv, d_mm) for uv, d_mm in best_cand.get_insertion_points_uvd()]
                c_3d_c_on_surface = self.projector.screen_to_camera(center, center_d_mm)
                length_to_center = max([pt.z for pt in insertion_points_c]) - c_3d_c_on_surface.z
                c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z + length_to_center)
                insertion_points_and_center_w = self.tf_client.transform_points(header, (*insertion_points_c, c_3d_c))
                insertion_points_w = insertion_points_and_center_w[:-1]
                c_3d_w = insertion_points_and_center_w[-1]
                c_orientation = self.pose_estimator.get_orientation(depth, mask)
                bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector, depth)

                # NOTE: unclockwise seen from image plane is positive in cnd.angle, so convert as rotate on z-axis
                # NOTE: 指位置が同じになる角度は複数存在するので候補に追加している
                angle = -best_cand.angle + self.hand_mount_rotation
                angles = []
                for i in range(1, self.finger_num + 1):
                    raw_rotated_angle = angle - (self.base_angle * i)
                    rotated_angle = raw_rotated_angle + 360 if raw_rotated_angle < -360 else raw_rotated_angle
                    reversed_rotated_angle = rotated_angle + 360
                    angles.extend([rotated_angle, reversed_rotated_angle])
                angles.sort(key=abs)
                objects.append(DetectedObject(
                    points=[pt.point for pt in insertion_points_w],
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

            self.visualize_client.visualize_candidates(vis_base_img_msg, candidates_list)
            if self.dbg_info_publisher:
                self.dbg_info_publisher.publish(GraspDetectionDebugInfo(header, candidates_list))
            spent = time() - start_time
            print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, objects: {len(objects)} ({len(instances)})")
            self.server.set_succeeded(GraspDetectionResult(header, objects))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    finger_num = rospy.get_param("finger_num")
    unit_angle = rospy.get_param("unit_angle")
    hand_radius_mm = rospy.get_param("hand_radius_mm")
    finger_radius_mm = rospy.get_param("finger_radius_mm")
    hand_mount_rotation = rospy.get_param("hand_mount_rotation")
    elements_th = rospy.get_param("elements_th")
    center_diff_th = rospy.get_param("center_diff_th")
    el_insertion_th = rospy.get_param("el_insertion_th")
    el_contact_th = rospy.get_param("el_contact_th")
    el_bw_depth_th = rospy.get_param("el_bw_depth_th")
    hand_mount_rotation = rospy.get_param("hand_mount_rotation")
    info_topic = rospy.get_param("image_info_topic")
    enable_depth_filter = rospy.get_param("enable_depth_filter")
    enable_candidate_filter = rospy.get_param("enable_candidate_filter")
    debug = rospy.get_param("debug")

    GraspDetectionServer(
        "grasp_detection_server",
        finger_num=finger_num,
        unit_angle=unit_angle,
        hand_radius_mm=hand_radius_mm,
        finger_radius_mm=finger_radius_mm,
        hand_mount_rotation=hand_mount_rotation,
        elements_th=elements_th,
        center_diff_th=center_diff_th,
        el_insertion_th=el_insertion_th,
        el_contact_th=el_contact_th,
        el_bw_depth_th=el_bw_depth_th,
        info_topic=info_topic,
        enable_depth_filter=enable_depth_filter,
        enable_candidate_filter=enable_candidate_filter,
        debug=debug
    )
    rospy.spin()
