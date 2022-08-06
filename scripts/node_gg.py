#!/usr/bin/env python3
import warnings
from os.path import expanduser
from pathlib import Path
from random import randint
from typing import Union

import cv2
import message_filters as mf
import numpy as np
import rospy
from cv_bridge import CvBridge
from detect.msg import DetectedObject, DetectedObjectsStamped, InstancesStamped
from geometry_msgs.msg import Point, PointStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from utils.grasp import generate_candidates_list
from utils.visualize import draw_candidates_and_boxes


def get_direction(cam_info, u, v):
    cam_model = PinholeCameraModel()
    cam_model.fromCameraInfo(cam_info)
    vector = np.array(cam_model.projectPixelTo3dRay((u, v)))
    return vector


def project_to_3d(cam_info, u, v, depth, frame_id, stamp, distance_margin=0):
    unit_v = get_direction(cam_info, u, v)
    distance = depth[u, v] / 1000 + distance_margin  # mm to m

    object_point = PointStamped(point=Point(*(unit_v * distance)))
    object_point.header.frame_id = frame_id
    object_point.header.stamp = stamp

    return object_point


def transform(tf_buffer: Buffer, point: PointStamped, target_frame: str, trans=None) -> PointStamped:
    if not trans:
        trans = tf_buffer.lookup_transform(
            target_frame, point.header.frame_id, point.header.stamp)
    tf_point = do_transform_point(
        point, trans)
    return tf_point


def callback(img_msg: Image, depth_msg: Image,
             instances_msg: InstancesStamped,
             callback_args: Union[list, tuple]):
    tf_buffer: Buffer = callback_args[0]
    cam_info = callback_args[1]
    monomask_publisher: rospy.Publisher = callback_args[2]
    cndsimg_publisher: rospy.Publisher = callback_args[3]
    objects_publisher: rospy.Publisher = callback_args[4]
    bridge = CvBridge()

    try:
        img = bridge.imgmsg_to_cv2(img_msg)
        depth = bridge.imgmsg_to_cv2(depth_msg)
        masks = np.array([bridge.imgmsg_to_cv2(x.mask)
                         for x in instances_msg.instances])
        # centers = np.array([x.center
        #                     for x in instances_msg.instances])
        # create indexed image
        indexed_img = np.zeros_like(masks[0])  # (h,w), 0 is bg
        for i, mask in enumerate(masks):
            # TODO: 物体の深度順にソートしたうえで結合したい
            indexed_img[mask == 1] = i + 1

        # centersどちら使うべきか
        candidates_list, contours, rotated_boxes, radiuses, centers = \
            generate_candidates_list(indexed_img, 20, 'min')
        # choice specific candidate
        target_indexes = [randint(0, len(candidates)-1)
                          for candidates in candidates_list]

        detected_objects_msg = DetectedObjectsStamped()
        detected_objects_msg.header.frame_id = "world"
        detected_objects_msg.header.stamp = instances_msg.header.stamp
        trans = tf_buffer.lookup_transform(
            "world", depth_msg.header.frame_id, depth_msg.header.stamp)
        for i in range(instances_msg.num_instances):
            center = centers[i]
            # TODO: get radius in meter | projectしたものの距離をとるべきか
            # radius = radiuses[i]
            p1, p2 = candidates_list[i][target_indexes[i]]

            p1_3d = project_to_3d(cam_info, int(p1[0]), int(p2[1]), depth,
                                  depth_msg.header.frame_id, depth_msg.header.stamp)
            p2_3d = project_to_3d(cam_info, int(p2[0]), int(p2[1]), depth,
                                  depth_msg.header.frame_id, depth_msg.header.stamp)
            radius = np.linalg.norm(
                np.array([p1_3d.point.x, p1_3d.point.y, p1_3d.point.z]) -
                np.array([p2_3d.point.x, p2_3d.point.y, p2_3d.point.z])
            ) / 2
            height = radius / 2
            center_3d = project_to_3d(cam_info, int(center[0]), int(center[1]), depth,
                                      depth_msg.header.frame_id, depth_msg.header.stamp,
                                      # height分ずらす
                                      distance_margin=height)

            rospy.loginfo(radius)

            detected_objects_msg.objects.append(
                DetectedObject(
                    p1=transform(tf_buffer, p1_3d, "world", trans),
                    p2=transform(tf_buffer, p2_3d, "world", trans),
                    center=transform(tf_buffer, center_3d, "world", trans),
                    radius=radius, height=height)
            )

        objects_publisher.publish(detected_objects_msg)

        res_img = np.where(indexed_img > 0, 255, 0)[
            :, :, np.newaxis].astype("uint8")
        res_img_msg = bridge.cv2_to_imgmsg(res_img, "mono8")

        res_img2 = img.copy()
        res_img2 = draw_candidates_and_boxes(
            res_img2, candidates_list, rotated_boxes, target_indexes=target_indexes, gray=True, copy=True)
        rospy.loginfo(f"{candidates_list}")
        res_img2_msg = bridge.cv2_to_imgmsg(res_img2, "rgb8")

        # outputs info publish
        header = Header(stamp=rospy.get_rostime(),
                        frame_id=img_msg.header.frame_id)
        res_img_msg.header = header
        monomask_publisher.publish(res_img_msg)
        cndsimg_publisher.publish(res_img2_msg)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rospy.init_node("grasp_candidates_node", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    fps = rospy.get_param(
        "fps", 10.)
    image_topics = rospy.get_param(
        "image_topic", "/body_camera/color/image_raw")
    # alignedとcolorで時間ずれてそうなので注意
    depth_topics = rospy.get_param(
        "depth_topic", "/body_camera/aligned_depth_to_color/image_raw")
    # depth_topics = rospy.get_param(
    #     "depth_topic", "/body_camera/depth/image_raw")
    instances_topics = rospy.get_param(
        "instances_topic", "/body_camera/color/image_raw/instances")
    info_topics = rospy.get_param(
        "info_topics", "/body_camera/aligned_depth_to_color/camera_info")

    delay = 1 / fps * 0.5
    # for instances_topic in instances_topics.split():
    instances_topic = instances_topics
    image_topic = image_topics
    depth_topic = depth_topics
    info_topic = info_topics

    # depth_topic = instances_topic.replace("color", "aligned_depth_to_color")
    rospy.loginfo(f"sub: {instances_topic}, {depth_topic}")
    monomask_publisher = rospy.Publisher(
        "/mono_mask", Image, queue_size=10)
    cndsimg_publisher = rospy.Publisher(
        "/condidates_img", Image, queue_size=10)
    objects_publisher = rospy.Publisher(
        "/detected_objects", DetectedObjectsStamped, queue_size=10)
    img_subscriber = mf.Subscriber(image_topic, Image)
    depth_subscriber = mf.Subscriber(depth_topic, Image)
    instances_subscriber = mf.Subscriber(instances_topic, InstancesStamped)

    subscribers = [img_subscriber, depth_subscriber, instances_subscriber]

    tf_buffer = Buffer()
    tf_lisner = TransformListener(tf_buffer)
    cam_info = rospy.wait_for_message(
        info_topic, CameraInfo, timeout=None)
    ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
    ts.registerCallback(callback, (tf_buffer, cam_info, monomask_publisher,
                        cndsimg_publisher, objects_publisher))

    rospy.spin()
