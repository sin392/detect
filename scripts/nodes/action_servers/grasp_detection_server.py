#!/usr/bin/env python3
from multiprocessing import Pool
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


def process_instance_segmentation_result_routine(depth, instance_msg, detect_func):
    instance_center = np.array(instance_msg.center)
    bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
    contour = multiarray2numpy(int, np.int32, instance_msg.contour)
    # detect candidates
    bbox_short_side_px, _ = bbox_handler.get_sides_2d()
    radius_for_augment = bbox_short_side_px // 2
    candidates = detect_func(center=instance_center, depth=depth, contour=contour, radius_for_augment=radius_for_augment)

    return (candidates, instance_center, bbox_handler)


def create_candidates_msg(original_center, valid_candidates, target_index):
    return Candidates(candidates=[
        Candidate(
            PointTuple2D(cnd.get_center_uv()),
            [PointTuple2D(pt) for pt in cnd.get_insertion_points_uv()],
            [PointTuple2D(pt) for pt in cnd.get_contact_points_uv()],
            cnd.total_score,
            cnd.is_valid
        )
        for cnd in valid_candidates
    ],
        # bbox=bbox_handler.msg,
        center=PointTuple2D(original_center),
        target_index=target_index
    )


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, unit_angle: int, hand_radius_mm: int, finger_radius_mm: int,
                 hand_mount_rotation: int, approach_coef: float,
                 elements_th: float, center_diff_th: float, el_insertion_th: float, el_contact_th: float, el_bw_depth_th: float,
                 info_topic: str, enable_depth_filter: bool, enable_candidate_filter: bool,
                 augment_anchors: bool, angle_for_augment: int,
                 debug: bool):
        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.unit_angle = unit_angle
        self.base_angle = 360 // finger_num
        self.hand_radius_mm = hand_radius_mm  # length between center and edge
        self.finger_radius_mm = finger_radius_mm
        self.hand_mount_rotation = hand_mount_rotation
        self.approach_coef = approach_coef
        self.elements_th = elements_th
        self.center_diff_th = center_diff_th
        self.el_insertion_th = el_insertion_th
        self.el_contact_th = el_contact_th
        self.el_bw_depth_th = el_bw_depth_th
        self.enable_candidate_filter = enable_candidate_filter
        self.augment_anchors = augment_anchors
        self.angle_for_augment = angle_for_augment
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
                                            el_bw_depth_th=el_bw_depth_th,
                                            augment_anchors=augment_anchors,
                                            angle_for_augment=angle_for_augment)

        self.pool = Pool(100)

        self.server = SimpleActionServer(name, GraspDetectionAction, self.callback, False)
        self.server.start()

    def depth_filtering(self, img_msg, depth_msg, img, depth, masks, n=5):
        if self.cdt_client:
            merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0).astype("uint8")
            merged_mask_msg = self.bridge.cv2_to_imgmsg(merged_mask)
            opt_depth_th = self.cdt_client.compute(depth_msg, merged_mask_msg, n=n)
            flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=n)
            flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
            vis_base_img_msg = self.bridge.cv2_to_imgmsg(flont_img)

            rospy.loginfo(opt_depth_th)
        else:
            opt_depth_th = UINT16MAX
            vis_base_img_msg = img_msg

        return opt_depth_th, vis_base_img_msg

    def compute_object_center_pose_stampd(self, depth, mask, c_3d_c_on_surface, header):
        c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z)
        c_3d_w = self.tf_client.transform_point(header, c_3d_c)
        c_orientation = self.pose_estimator.get_orientation(depth, mask)

        return PoseStamped(
            Header(frame_id="base_link"),
            Pose(
                position=c_3d_w.point,
                orientation=c_orientation
            )
        )

    def compute_approach_distance(self, c_3d_c_on_surface, insertion_points_c):
        bottom_z = max([pt.z for pt in insertion_points_c])
        top_z = c_3d_c_on_surface.z
        # length_to_center = bottom_z - top_z
        length_to_center = (bottom_z - top_z) * self.approach_coef  # ??????????????????????????????????????????????????????
        return length_to_center

    def compute_object_3d_radiuses(self, depth, bbox_handler):
        bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector, depth)
        short_raidus = bbox_short_side_3d / 2
        long_radius = bbox_long_side_3d / 2
        return (short_raidus, long_radius)

    def augment_angles(self, angle):
        angles = []
        for i in range(1, self.finger_num + 1):
            raw_rotated_angle = angle - (self.base_angle * i)
            rotated_angle = raw_rotated_angle + 360 if raw_rotated_angle < -360 else raw_rotated_angle
            reversed_rotated_angle = rotated_angle + 360
            angles.extend([rotated_angle, reversed_rotated_angle])
        angles.sort(key=abs)
        return angles

    def callback(self, goal: GraspDetectionGoal):
        print("receive request")
        img_msg = goal.image
        depth_msg = goal.depth
        points_msg = goal.points
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp
        header = Header(frame_id=frame_id, stamp=stamp)
        try:
            start_time = time()
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg)
            # TODO: depth?????????????????????????????????merged_mask??????????????????????????????????????????
            masks = [self.bridge.imgmsg_to_cv2(instance_msg.mask) for instance_msg in instances]
            # TODO: compute n by camera distance
            opt_depth_th, vis_base_img_msg = self.depth_filtering(img_msg, depth_msg, img, depth, masks, n=5)

            objects: List[DetectedObject] = []  # ?????????????????????
            candidates_list: List[Candidates] = []  # ?????????????????????

            # ????????????????????? (????????????)
            routine_args = []
            for instance_msg, mask in zip(instances, masks):
                # ignore other than instances are located on top of stacks
                # TODO: ?????????????????????????????????????????????????????????????????????????????????????????????
                instance_min_d = depth[mask > 0].min()
                if instance_min_d > opt_depth_th:
                    continue
                routine_args.append((depth, instance_msg, self.grasp_detector.detect))
            results = self.pool.starmap(process_instance_segmentation_result_routine, routine_args)

            # TODO: ???????????????????????????????????????
            for obj_index, (candidates, instance_center, bbox_handler) in enumerate(results):
                if len(candidates) == 0:
                    continue
                # select best candidate
                valid_candidates = [cnd for cnd in candidates if cnd.is_valid] if enable_candidate_filter else candidates
                is_valid = len(valid_candidates) > 0
                valid_scores = [cnd.total_score for cnd in valid_candidates]
                target_index = np.argmax(valid_scores) if is_valid else 0

                # candidates_list ???????????????????????????????????????
                candidates_list.append(create_candidates_msg(instance_center, valid_candidates, target_index))

                # TODO: is_framein?????????????????????????????????
                include_any_frameout = not np.any([cnd.is_framein for cnd in valid_candidates])
                if include_any_frameout or not is_valid:
                    continue

                best_cand = valid_candidates[target_index]
                # 3d projection
                insertion_points_c = [self.projector.screen_to_camera_2(points_msg, uv) for uv in best_cand.get_insertion_points_uv()]
                c_3d_c_on_surface = self.projector.screen_to_camera_2(points_msg, best_cand.get_center_uv())
                # insertion_points_c = [self.projector.screen_to_camera(uv, d_mm) for uv, d_mm in best_cand.get_insertion_points_uvd()]
                # c_3d_c_on_surface = self.projector.screen_to_camera(*best_cand.get_center_uvd())
                # compute approach distance
                length_to_center = self.compute_approach_distance(c_3d_c_on_surface, insertion_points_c)
                # compute center pose stamped (world coords)
                insertion_points_msg = [pt.point for pt in self.tf_client.transform_points(header, insertion_points_c)]
                center_pose_stamped_msg = self.compute_object_center_pose_stampd(depth, mask, c_3d_c_on_surface, header)
                # compute 3d radiuses
                short_radius_3d, long_radius_3d = self.compute_object_3d_radiuses(depth, bbox_handler)
                # angle?????????????????????
                # NOTE: unclockwise seen from image plane is positive in cnd.angle, so convert as rotate on z-axis
                # NOTE: ???????????????????????????????????????????????????????????????????????????????????????
                angle = -best_cand.angle + self.hand_mount_rotation
                # ????????????(????????????????????????)?????????????????????????????????
                angles = self.augment_angles(angle)
                objects.append(DetectedObject(
                    points=insertion_points_msg,
                    center_pose=center_pose_stamped_msg,
                    angles=angles,
                    short_radius=short_radius_3d,
                    long_radius=long_radius_3d,
                    length_to_center=length_to_center,
                    score=best_cand.total_score,
                    index=obj_index  # for visualize
                )
                )

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
    approach_coef = rospy.get_param("approach_coef")
    elements_th = rospy.get_param("elements_th")
    center_diff_th = rospy.get_param("center_diff_th")
    el_insertion_th = rospy.get_param("el_insertion_th")
    el_contact_th = rospy.get_param("el_contact_th")
    el_bw_depth_th = rospy.get_param("el_bw_depth_th")
    info_topic = rospy.get_param("image_info_topic")
    enable_depth_filter = rospy.get_param("enable_depth_filter")
    enable_candidate_filter = rospy.get_param("enable_candidate_filter")
    augment_anchors = rospy.get_param("augment_anchors")
    angle_for_augment = rospy.get_param("angle_for_augment")
    debug = rospy.get_param("debug")

    GraspDetectionServer(
        "grasp_detection_server",
        finger_num=finger_num,
        unit_angle=unit_angle,
        hand_radius_mm=hand_radius_mm,
        finger_radius_mm=finger_radius_mm,
        hand_mount_rotation=hand_mount_rotation,
        approach_coef=approach_coef,
        elements_th=elements_th,
        center_diff_th=center_diff_th,
        el_insertion_th=el_insertion_th,
        el_contact_th=el_contact_th,
        el_bw_depth_th=el_bw_depth_th,
        info_topic=info_topic,
        enable_depth_filter=enable_depth_filter,
        enable_candidate_filter=enable_candidate_filter,
        augment_anchors=augment_anchors,
        angle_for_augment=angle_for_augment,
        debug=debug
    )
    rospy.spin()
