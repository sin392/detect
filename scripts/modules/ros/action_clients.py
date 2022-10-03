#!/usr/bin/env python3
from typing import List

from actionlib import SimpleActionClient
from detect.msg import (Candidates, DetectedObject, GraspDetectionAction,
                        GraspDetectionGoal, Instance,
                        InstanceSegmentationAction, InstanceSegmentationGoal,
                        TransformPointAction, TransformPointGoal,
                        VisualizeCandidatesAction, VisualizeCandidatesGoal)
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class TFClient(SimpleActionClient):
    def __init__(self, target_frame: str, ns="tf_transform_server", ActionSpec=TransformPointAction):
        super().__init__(ns, ActionSpec)
        self.target_frame = target_frame

        self.wait_for_server()

    def transform_point(self, header: Header, point: Point) -> PointStamped:
        # 同期的だからServiceで実装してもよかったかも
        self.send_goal_and_wait(TransformPointGoal(
            self.target_frame, PointStamped(header, point)))
        result = self.get_result().result
        # get_resultのresultのheaderは上書きしないと固定値？
        return result

    def transform_points(self, header: Header, points: List[Point]) -> List[PointStamped]:
        result = [self.transform_point(header, point) for point in points]
        return result


class VisualizeClient(SimpleActionClient):
    def __init__(self, ns="visualize_server", ActionSpec=VisualizeCandidatesAction):
        super().__init__(ns, ActionSpec)
        self.stack = []
        self.wait_for_server()

    def visualize_candidates(self, base_image: Image, candidates_list: List[Candidates]):
        self.send_goal(VisualizeCandidatesGoal(base_image, candidates_list))

    def push_item(self, candidates: Candidates):
        self.stack.append(candidates)

    def visualize_stacked_candidates(self, base_image: Image):
        self.send_goal(VisualizeCandidatesGoal(base_image, self.stack))
        self.clear_stack()

    def clear_stack(self):
        self.stack = []


class InstanceSegmentationClient(SimpleActionClient):
    def __init__(self, ns="instance_segmentation_server", ActionSpec=InstanceSegmentationAction):
        super().__init__(ns, ActionSpec)
        self.wait_for_server()

    def predict(self, image: Image) -> List[Instance]:
        self.send_goal_and_wait(InstanceSegmentationGoal(image))
        res = self.get_result().instances
        return res


class GraspDetectionClient(SimpleActionClient):
    def __init__(self, ns="grasp_detection_server", ActionSpec=GraspDetectionAction):
        super().__init__(ns, ActionSpec)
        self.stack = []
        self.wait_for_server()

    def detect(self, image: Image, depth: Image) -> List[DetectedObject]:
        self.send_goal_and_wait(GraspDetectionGoal(image, depth))
        res = self.get_result().objects
        return res
