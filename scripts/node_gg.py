#!/usr/bin/env python3
from random import randint
from typing import List, Tuple

import message_filters as mf
import numpy as np
import rospy
from cv_bridge import CvBridge
from detect.msg import Candidate, Candidates, Instance, InstancesStamped
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from modules.const import FRAME_SIZE
from modules.grasp import ParallelGraspDetector
from modules.ros.action_client import TFClient, VisualizeClient
from modules.ros.publisher import DetectedObjectsPublisher
from modules.ros.utils import (PointProjector, PoseEstimator, bboxmsg2list,
                               multiarray2numpy)

CallbackArgsType = Tuple[CvBridge,
                         DetectedObjectsPublisher, PointProjector,
                         PoseEstimator, ParallelGraspDetector,
                         TFClient, VisualizeClient]


def callback(img_msg: Image, depth_msg: Image,
             instances_msg: InstancesStamped,
             callback_args: CallbackArgsType):
    bridge = callback_args[0]
    objects_publisher = callback_args[1]
    projector = callback_args[2]
    pose_estimator = callback_args[3]
    grasp_detector = callback_args[4]
    tf_client = callback_args[5]
    visualize_client = callback_args[6]

    frame_id = depth_msg.header.frame_id
    stamp = depth_msg.header.stamp
    header = Header(frame_id=frame_id, stamp=stamp)
    try:
        depth = bridge.imgmsg_to_cv2(depth_msg)

        instances: List[Instance] = instances_msg.instances
        candidates_list: List[Candidates] = []
        for instance_msg in instances:
            center = instance_msg.center
            bbox_msg = instance_msg.bbox
            bbox = bboxmsg2list(bbox_msg)
            contour = multiarray2numpy(int, np.int32, instance_msg.contour)
            mask = bridge.imgmsg_to_cv2(instance_msg.mask)

            candidates = grasp_detector.detect(
                center, bbox, contour, depth, filter=True)
            if len(candidates) == 0:
                continue

            # select best candidate
            target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
            candidates_list.append(Candidates([Candidate(*p1, *p2) for p1, p2 in candidates], bbox_msg, target_index))

            # 3d projection
            p1_3d_c, p2_3d_c = [projector.pixel_to_3d(*point[::-1], depth) for point in candidates[target_index]]
            long_radius = np.linalg.norm(
                np.array([p1_3d_c.x, p1_3d_c.y, p1_3d_c.z]) - np.array([p2_3d_c.x, p2_3d_c.y, p2_3d_c.z])
            ) / 2
            short_radius = long_radius / 2

            c_3d_c = projector.pixel_to_3d(
                *center[::-1],
                depth,
                margin_mm=short_radius  # 中心点は物体表面でなく中心座標を取得したいのでmargin_mmを指定
            )

            # transform from camera to world
            p1_3d_w, p2_3d_w, c_3d_w = tf_client.transform_points(header, (p1_3d_c, p2_3d_c, c_3d_c))

            c_orientation = pose_estimator.get_orientation(depth, mask)

            objects_publisher.push_item(
                p1=p1_3d_w.point,
                p2=p2_3d_w.point,
                center_pose=Pose(
                    position=c_3d_w.point,
                    orientation=c_orientation
                ),
                short_radius=short_radius,
                long_radius=long_radius,
            )

        objects_publisher.publish_stack("base_link", stamp)
        visualize_client.visualize_candidates(img_msg, candidates_list)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    rospy.init_node("grasp_candidates_node", log_level=rospy.INFO)

    fps = rospy.get_param("fps")
    delay = 1 / fps  # * 0.5

    image_topics = rospy.get_param("image_topic")
    depth_topics = rospy.get_param("depth_topic")
    instances_topics = rospy.get_param("instances_topic")
    info_topic = rospy.get_param("depth_info_topic")

    # for instances_topic in instances_topics.split():
    instances_topic = instances_topics
    image_topic = image_topics
    depth_topic = depth_topics
    info_topic = info_topic

    tf_client = TFClient("base_link")
    visualize_client = VisualizeClient()

    rospy.loginfo(f"sub: {instances_topic}, {depth_topic}")
    # Publishers
    objects_publisher = DetectedObjectsPublisher(
        "/detected_objects", queue_size=10)
    # Subscribers
    img_subscriber = mf.Subscriber(image_topic, Image)
    depth_subscriber = mf.Subscriber(depth_topic, Image)
    instances_subscriber = mf.Subscriber(instances_topic, InstancesStamped)
    subscribers = [img_subscriber, depth_subscriber, instances_subscriber]

    cam_info = rospy.wait_for_message(
        info_topic, CameraInfo, timeout=None)

    bridge = CvBridge()
    projector = PointProjector(cam_info)
    pose_estimator = PoseEstimator()
    grasp_detector = ParallelGraspDetector(
        frame_size=FRAME_SIZE, unit_angle=15, margin=3, func="min")

    callback_args: CallbackArgsType = (
        bridge, objects_publisher, projector, pose_estimator,
        grasp_detector, tf_client, visualize_client)
    ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
    ts.registerCallback(callback, callback_args)

    rospy.spin()
