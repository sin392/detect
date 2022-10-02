#!/usr/bin/env python3
from random import randint

import message_filters as mf
import numpy as np
import rospy
from cv_bridge import CvBridge
from detect.msg import Candidate, Candidates
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from modules.const import FRAME_SIZE
from modules.grasp import ParallelGraspDetector
from modules.ros.action_client import (InstanceSegmentationClient, TFClient,
                                       VisualizeClient)
from modules.ros.msg import RotatedBoundingBox
from modules.ros.publisher import DetectedObjectsPublisher
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy


class GraspGenerationNode:
    def __init__(self, name: str, fps: float, image_topic: str, depth_topic: str, objects_topic: str, info_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        delay = 1 / fps  # * 0.5
        cam_info = rospy.wait_for_message(
            info_topic, CameraInfo, timeout=None)

        # Publishers
        self.objects_publisher = DetectedObjectsPublisher(objects_topic, queue_size=10)
        # Subscribers
        img_subscriber = mf.Subscriber(image_topic, Image)
        depth_subscriber = mf.Subscriber(depth_topic, Image)
        subscribers = [img_subscriber, depth_subscriber]
        # Action Clients
        self.is_client = InstanceSegmentationClient()
        self.tf_client = TFClient("base_link")
        self.visualize_client = VisualizeClient()
        # Others
        self.bridge = CvBridge()
        self.projector = PointProjector(cam_info)
        self.pose_estimator = PoseEstimator()
        self.grasp_detector = ParallelGraspDetector(
            frame_size=FRAME_SIZE, unit_angle=15, margin=3, func="min")

        self.ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
        self.ts.registerCallback(self.callback)

    def callback(self, img_msg: Image, depth_msg: Image):
        frame_id = depth_msg.header.frame_id
        stamp = depth_msg.header.stamp
        header = Header(frame_id=frame_id, stamp=stamp)
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg)
            for instance_msg in instances:
                center = instance_msg.center
                bbox_msg = instance_msg.bbox
                bbox = RotatedBoundingBox.tolist(bbox_msg)
                contour = multiarray2numpy(int, np.int32, instance_msg.contour)
                mask = self.bridge.imgmsg_to_cv2(instance_msg.mask)

                candidates = self.grasp_detector.detect(
                    center, bbox, contour, depth, filter=True)
                if len(candidates) == 0:
                    continue

                # select best candidate
                target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
                self.visualize_client.push_item(Candidates([Candidate(*p1, *p2) for p1, p2 in candidates], bbox_msg, target_index))

                # 3d projection
                p1_3d_c, p2_3d_c = [self.projector.pixel_to_3d(*point[::-1], depth) for point in candidates[target_index]]
                long_radius = np.linalg.norm(
                    np.array([p1_3d_c.x, p1_3d_c.y, p1_3d_c.z]) - np.array([p2_3d_c.x, p2_3d_c.y, p2_3d_c.z])
                ) / 2
                short_radius = long_radius / 2

                c_3d_c = self.projector.pixel_to_3d(
                    *center[::-1],
                    depth,
                    margin_mm=short_radius  # 中心点は物体表面でなく中心座標を取得したいのでmargin_mmを指定
                )

                # transform from camera to world
                p1_3d_w, p2_3d_w, c_3d_w = self.tf_client.transform_points(header, (p1_3d_c, p2_3d_c, c_3d_c))

                c_orientation = self.pose_estimator.get_orientation(depth, mask)

                self.objects_publisher.push_item(
                    p1=p1_3d_w.point,
                    p2=p2_3d_w.point,
                    center_pose=Pose(
                        position=c_3d_w.point,
                        orientation=c_orientation
                    ),
                    short_radius=short_radius,
                    long_radius=long_radius,
                )

            self.objects_publisher.publish_stack("base_link", stamp)
            self.visualize_client.visualize_stacked_candidates(img_msg)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    fps = rospy.get_param("fps")
    image_topic = rospy.get_param("image_topic")
    depth_topic = rospy.get_param("depth_topic")
    objects_topic = rospy.get_param("objects_topic")
    info_topic = rospy.get_param("depth_info_topic")

    GraspGenerationNode(
        "grasp_generation_node",
        fps=fps,
        image_topic=image_topic,
        depth_topic=depth_topic,
        objects_topic=objects_topic,
        info_topic=info_topic
    )

    rospy.spin()
