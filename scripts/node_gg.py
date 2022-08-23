#!/usr/bin/env python3
import warnings
from os.path import expanduser
from pathlib import Path
from random import randint
from typing import List, Union

import cv2
import message_filters as mf
import numpy as np
import rospy
from cv_bridge import CvBridge
from detect.msg import (DetectedObject, DetectedObjectsStamped, Instance,
                        InstancesStamped, RotatedBoundingBox)
from geometry_msgs.msg import (Point, PointStamped, Pose, PoseStamped,
                               Quaternion)
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from tf.transformations import quaternion_from_matrix

from entities.image import IndexedMask
from ros.publisher import ImageMatPublisher
from ros.utils import multiarray2numpy
from utils.grasp import ParallelGraspDetector, generate_candidates_list
from utils.visualize import convert_rgb_to_3dgray, draw_bbox, draw_candidates


def bboxmsg2list(msg: RotatedBoundingBox):
    return np.int0([msg.upper_left, msg.upper_right, msg.lower_right, msg.lower_left])


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


def get_orientation(u, v, depth, mask):
    pca = PCA(n_components=3)
    ss = StandardScaler()
    # ここの値あってるか要検証...
    pts = [(u, v, depth[y, x]) for y, x in zip(*np.where(mask == 1))]
    pca.fit(ss.fit_transform(pts))
    n, t, b = pca.components_
    rmat_44 = np.eye(4)
    rmat_33 = np.dstack([n, t, b])[0]
    rmat_44[:3, :3] = rmat_33
    # 4x4回転行列しか受け入れない罠
    q = quaternion_from_matrix(rmat_44)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def transform_point(tf_buffer: Buffer, point: PointStamped, target_frame: str, trans=None) -> PointStamped:
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
    # monomask_publisher: ImageMatPublisher = callback_args[2]
    cndsimg_publisher: ImageMatPublisher = callback_args[3]
    objects_publisher: rospy.Publisher = callback_args[4]
    bridge = CvBridge()

    try:
        img = bridge.imgmsg_to_cv2(img_msg)
        depth = bridge.imgmsg_to_cv2(depth_msg)
        # TODO: 物体の深度順にソートできてる？
        # masksは移行したい
        masks = np.array([bridge.imgmsg_to_cv2(x.mask)
                         for x in instances_msg.instances])
        indexed_img = IndexedMask(masks)

        # bboxはすでに計算済みだが、rotated_bboxの計算と重複してる...
        candidates_list, _, rotated_boxes, radiuses, _ = \
            generate_candidates_list(indexed_img, 20, 3, 'min')
        # choice specific candidate
        # detected_objects_msg = DetectedObjectsStamped()
        # detected_objects_msg.header.frame_id = "world"
        # detected_objects_msg.header.stamp = instances_msg.header.stamp
        # trans = tf_buffer.lookup_transform(
        #     "world", depth_msg.header.frame_id, depth_msg.header.stamp)
        assert instances_msg.num_instances == len(candidates_list)
        target_indexes = []
        cnds_img = convert_rgb_to_3dgray(img)
        instances: List[Instance] = instances_msg.instances
        grasp_detector = ParallelGraspDetector(15, margin=3, func="min")
        for i, instance in enumerate(instances):
            center = instance.center
            bbox = bboxmsg2list(instance.bbox)
            contour = multiarray2numpy(int, np.int32, instance.contour)
            candidates = grasp_detector.detect(center, bbox, contour, masks[i])

            u, v = int(center[1]), int(center[0])
            # radius = radiuses[i]

            # filter candidates, is x,y? or h,w?
            # 把持点のdepthがcenterのdepthより大きくないとだめ
            candidates = [(p1, p2) for p1, p2 in candidates
                          if min(depth[int(p1[1])][int(p1[0])], depth[int(p2[1])][int(p2[0])]) >= depth[u, v]]
            rospy.loginfo(f"{len(candidates_list[i])} {len(candidates)}")
            if len(candidates) == 0:
                rospy.loginfo("skip")
                continue
            candidates_list[i] = candidates

            target_index = randint(
                0, len(candidates)-1) if len(candidates) != 0 else 0
            p1, p2 = candidates[target_index]
            target_indexes.append(target_index)

            p1_3d = project_to_3d(cam_info, int(p1[1]), int(p1[0]), depth,
                                  depth_msg.header.frame_id, depth_msg.header.stamp)
            p2_3d = project_to_3d(cam_info, int(p2[1]), int(p2[0]), depth,
                                  depth_msg.header.frame_id, depth_msg.header.stamp)
            radius = np.linalg.norm(
                np.array([p1_3d.point.x, p1_3d.point.y, p1_3d.point.z]) -
                np.array([p2_3d.point.x, p2_3d.point.y, p2_3d.point.z])
            ) / 2
            height = radius / 2
            center_3d = project_to_3d(cam_info, u, v, depth,
                                      depth_msg.header.frame_id, depth_msg.header.stamp,
                                      # height分ずらす
                                      distance_margin=height)
            center_orientation = get_orientation(u, v, depth, masks[i])

            cnds_img = draw_bbox(cnds_img, bbox)
            cnds_img = draw_candidates(
                cnds_img, candidates, target_index=target_index)

        #     detected_objects_msg.objects.append(
        #         DetectedObject(
        #             radius=radius,
        #             height=height,
        #             p1=transform_point(tf_buffer, p1_3d, "world", trans),
        #             p2=transform_point(tf_buffer, p2_3d, "world", trans),
        #             center=PoseStamped(
        #                 header=Header(frame_id="world",
        #                               stamp=depth_msg.header.stamp),
        #                 pose=Pose(
        #                     # position is Point (not PointStamped)
        #                     position=transform_point(
        #                         tf_buffer, center_3d, "world", trans).point,
        #                     orientation=center_orientation
        #                 )
        #             )
        #         ))

        # objects_publisher.publish(detected_objects_msg)

        header = Header(stamp=rospy.get_rostime(),
                        frame_id=img_msg.header.frame_id)
        # monomask_publisher.publish(monomask, header=header)
        # monomask = np.where(indexed_img > 0, 255, 0)[
        #     :, :, np.newaxis].astype("uint8")
        cndsimg_publisher.publish(cnds_img, header=header)

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
    # Publishers
    monomask_publisher = ImageMatPublisher(
        "/mono_mask", queue_size=10)
    cndsimg_publisher = ImageMatPublisher(
        "/candidates_img", queue_size=10)
    objects_publisher = rospy.Publisher(
        "/detected_objects", DetectedObjectsStamped, queue_size=10)
    # Subscribers
    img_subscriber = mf.Subscriber(image_topic, Image)
    depth_subscriber = mf.Subscriber(depth_topic, Image)
    instances_subscriber = mf.Subscriber(instances_topic, InstancesStamped)
    subscribers = [img_subscriber, depth_subscriber, instances_subscriber]

    tf_buffer = Buffer()
    # tf_lisner = TransformListener(tf_buffer)
    cam_info = rospy.wait_for_message(
        info_topic, CameraInfo, timeout=None)
    ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
    ts.registerCallback(callback, (tf_buffer, cam_info, monomask_publisher,
                        cndsimg_publisher, objects_publisher))

    rospy.spin()
