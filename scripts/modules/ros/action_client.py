#!/usr/bin/env python3
from typing import List

from actionlib import SimpleActionClient
from detect.msg import (Candidates, TransformPointAction, TransformPointGoal,
                        VisualizeCandidatesAction, VisualizeCandidatesGoal)
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class TFClient(SimpleActionClient):
    def __init__(self, target_frame: str, ns="tf_transform", ActionSpec=TransformPointAction):
        super().__init__(ns, ActionSpec)
        self.target_frame = target_frame
        self.source_header = Header()

        self.wait_for_server()

    def set_source_header(self, header: Header):
        self.source_header = header

    def transform_point(self, point: Point) -> PointStamped:
        # 同期的だからServiceで実装してもよかったかも
        self.send_goal_and_wait(TransformPointGoal(self.target_frame, PointStamped(self.source_header, point)))
        result = self.get_result().result
        # get_resultのresultのheaderは上書きしないと固定値？
        result.header.frame_id = self.target_frame
        result.header.stamp = self.source_header.stamp
        return result

    def transform_points(self, header: Header, points: List[Point]) -> List[PointStamped]:
        self.set_source_header(header)
        result = [self.transform_point(point) for point in points]
        return result


class VisualizeClient(SimpleActionClient):
    def __init__(self, ns="visualize", ActionSpec=VisualizeCandidatesAction):
        super().__init__(ns, ActionSpec)
        self.wait_for_server()

    def visualize_candidates(self, base_image: Image, candidates_list: List[Candidates]):
        self.send_goal(VisualizeCandidatesGoal(base_image, candidates_list))
