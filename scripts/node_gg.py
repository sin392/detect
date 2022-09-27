#!/usr/bin/env python3
import warnings
from os.path import expanduser
from pathlib import Path
from random import randint
from typing import List, Tuple

import message_filters as mf
import numpy as np
import rospy
from actionlib import SimpleActionClient
from cv_bridge import CvBridge
from detect.msg import (DetectedObjectsStamped, Instance, InstancesStamped,
                        RotatedBoundingBox, TransformPointAction,
                        TransformPointGoal)
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix

from modules.grasp import ParallelGraspDetector
from modules.ros.publisher import DetectedObjectsPublisher, ImageMatPublisher
from modules.ros.utils import multiarray2numpy
from modules.visualize import convert_rgb_to_3dgray, draw_bbox, draw_candidates

FRAME_SIZE = (480, 640)


def bboxmsg2list(msg: RotatedBoundingBox):
    return np.int0([msg.upper_left, msg.upper_right, msg.lower_right, msg.lower_left])


class PointProjector:
    def __init__(self, cam_info):
        self.cam_info = cam_info

    def pixel_to_3d(self, u, v, depth, margin_mm=0) -> Point:
        """
        ピクセルをカメラ座標系へ３次元投影
        ---
        u,v: ピクセル位置
        depth: 深度画像
        margin_mm: 物体表面から中心までの距離[mm]
        """
        unit_v = self._get_direction(u, v)
        distance = depth[u, v] / 1000 + margin_mm  # mm to m
        object_point = Point(*(unit_v * distance))
        return object_point

    def _get_direction(self, u, v):
        """カメラ座標系原点から対象点までの方向ベクトルを算出"""
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(self.cam_info)
        vector = np.array(cam_model.projectPixelTo3dRay((u, v)))
        return vector


class PoseEstimator:
    def __init__(self):
        self.pca = PCA(n_components=3)
        self.ss = StandardScaler()

    def get_orientation(self, depth, mask) -> Quaternion:
        """マスクに重なったデプスからインスタンスの姿勢を算出"""
        # ここの値あってるか要検証...
        pts = [(x, y, depth[y, x]) for y, x in zip(*np.where(mask > 0))]
        self.pca.fit(self.ss.fit_transform(pts))
        n, t, b = self.pca.components_
        rmat_44 = np.eye(4)
        rmat_33 = np.dstack([n, t, b])[0]
        rmat_44[:3, :3] = rmat_33
        # 4x4回転行列しか受け入れない罠
        q = quaternion_from_matrix(rmat_44)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class CoordinateTransformer:
    def __init__(self, target_frame: str, client: SimpleActionClient):
        self.target_frame = target_frame
        self.client = client
        self.source_header = Header()

    def set_source_header(self, header: Header):
        self.sorce_header = header

    def transform_point(self, point: Point) -> PointStamped:
        # 同期的だからServiceで実装してもよかったかも
        self.client.send_goal_and_wait(TransformPointGoal(self.target_frame, PointStamped(self.source_header, point)))
        result = tf_client.get_result().result
        # get_resultのresultのheaderは上書きしないと固定値？
        result.header.frame_id = self.target_frame
        result.header.stamp = self.source_header.stamp
        return result


CallbackArgsType = Tuple[ImageMatPublisher, ImageMatPublisher,
                         DetectedObjectsPublisher, PointProjector,
                         PoseEstimator, ParallelGraspDetector,
                         CoordinateTransformer]


def callback(img_msg: Image, depth_msg: Image,
             instances_msg: InstancesStamped,
             callback_args: CallbackArgsType):
    bridge = CvBridge()
    # monomask_publisher: ImageMatPublisher = callback_args[0]
    cndsimg_publisher = callback_args[1]
    objects_publisher = callback_args[2]
    projector = callback_args[3]
    pose_estimator = callback_args[4]
    grasp_detector = callback_args[5]
    coords_transformer = callback_args[6]

    # detector should be moved outside callback
    frame_id = depth_msg.header.frame_id
    stamp = depth_msg.header.stamp
    header = Header(frame_id=frame_id, stamp=stamp)
    try:
        img = bridge.imgmsg_to_cv2(img_msg)
        depth = bridge.imgmsg_to_cv2(depth_msg)

        target_indexes = []
        cnds_img = convert_rgb_to_3dgray(img)
        instances: List[Instance] = instances_msg.instances
        for instance in instances:
            center = instance.center
            bbox = bboxmsg2list(instance.bbox)
            contour = multiarray2numpy(int, np.int32, instance.contour)
            mask = bridge.imgmsg_to_cv2(instance.mask)

            candidates = grasp_detector.detect(
                center, bbox, contour, depth, filter=True)
            if len(candidates) == 0:
                continue

            # select best candidate
            target_index = randint(0, len(candidates) - 1) if len(candidates) != 0 else 0
            p1, p2 = candidates[target_index]
            target_indexes.append(target_index)

            # 3d projection
            p1_3d_c = projector.pixel_to_3d(*p1[::-1], depth)
            p2_3d_c = projector.pixel_to_3d(*p2[::-1], depth)
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
            coords_transformer.set_source_header(header)
            p1_3d_w = coords_transformer.transform_point(p1_3d_c)
            p2_3d_w = coords_transformer.transform_point(p2_3d_c)
            c_3d_w = coords_transformer.transform_point(c_3d_c)

            c_orientation = pose_estimator.get_orientation(depth, mask)

            cnds_img = draw_bbox(cnds_img, bbox)
            cnds_img = draw_candidates(
                cnds_img, candidates, target_index=target_index)

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

        # monomask_publisher.publish(monomask, frame_id, stamp)
        # monomask = np.where(indexed_img > 0, 255, 0)[
        #     :, :, np.newaxis].astype("uint8")
        cndsimg_publisher.publish(cnds_img, frame_id, stamp)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rospy.init_node("grasp_candidates_node", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    fps = rospy.get_param(
        "fps", 30.)
    image_topics = rospy.get_param(
        "image_topic", "/myrobot/body_camera/color/image_raw")
    # alignedとcolorで時間ずれてそうなので注意
    depth_topics = rospy.get_param(
        "depth_topic", "/myrobot/body_camera/aligned_depth_to_color/image_raw")
    # depth_topics = rospy.get_param(
    #     "depth_topic", "/myrobot/body_camera/depth/image_raw")
    instances_topics = rospy.get_param(
        "instances_topic", "/myrobot/body_camera/color/image_raw/instances")
    info_topic = rospy.get_param(
        "depth_info_topic", "/myrobot/body_camera/aligned_depth_to_color/camera_info")

    delay = 1 / fps  # * 0.5
    # for instances_topic in instances_topics.split():
    instances_topic = instances_topics
    image_topic = image_topics
    depth_topic = depth_topics
    info_topic = info_topic

    tf_client = SimpleActionClient("tf_transform", TransformPointAction)
    tf_client.wait_for_server()

    # depth_topic = instances_topic.replace("color", "aligned_depth_to_color")
    rospy.loginfo(f"sub: {instances_topic}, {depth_topic}")
    # Publishers
    monomask_publisher = ImageMatPublisher(
        "/mono_mask", queue_size=10)
    cndsimg_publisher = ImageMatPublisher(
        "/candidates_img", queue_size=10)
    objects_publisher = DetectedObjectsPublisher(
        "/detected_objects", DetectedObjectsStamped, queue_size=10)
    # Subscribers
    img_subscriber = mf.Subscriber(image_topic, Image)
    depth_subscriber = mf.Subscriber(depth_topic, Image)
    instances_subscriber = mf.Subscriber(instances_topic, InstancesStamped)
    subscribers = [img_subscriber, depth_subscriber, instances_subscriber]

    cam_info = rospy.wait_for_message(
        info_topic, CameraInfo, timeout=None)

    projector = PointProjector(cam_info)
    pose_estimator = PoseEstimator()
    grasp_detector = ParallelGraspDetector(
        frame_size=FRAME_SIZE, unit_angle=15, margin=3, func="min")
    coords_transformer = CoordinateTransformer("base_link", tf_client)

    callback_args: CallbackArgsType = (
        monomask_publisher, cndsimg_publisher,
        objects_publisher, projector, pose_estimator,
        grasp_detector, coords_transformer)
    ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
    ts.registerCallback(callback, callback_args)

    rospy.spin()
