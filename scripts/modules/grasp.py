from typing import List, Optional, Tuple

import cv2
import numpy as np


class ParallelCandidate:
    def __init__(self, p1, p2, pc, depth, h, w, contour, finger_radius):
        # TODO: フィルタリングに関しては1点のdepthではなく半径finger_radius内の領域のdepthの平均をとるべきかも
        self.h = h
        self.w = w
        self.contour = contour
        self.depth = depth
        self.finger_radius = finger_radius
        self.p1_u, self.p1_v, self.p1_d = self._format(p1)
        self.p2_u, self.p2_v, self.p2_d = self._format(p2)
        self.pc_u, self.pc_v, self.pc_d = self._format(pc)
        self.is_valid = self.validate()

    def _format(self, pt) -> Tuple[int, int, float]:
        u = int(pt[0].item())
        v = int(pt[1].item())
        d = self.depth[v][u]  # [mm]
        return (u, v, d)

    def validate(self):
        is_invalid = False
        is_invalid = is_invalid or self._is_outside_frame()
        is_invalid = is_invalid or self._is_depth_missing()
        is_invalid = is_invalid or self._is_in_mask()
        is_invalid = is_invalid or self._is_center_under_edges()
        return not is_invalid

    def get_candidate_points_on_rgbd(self):
        points = (((self.p1_u, self.p1_v), self.p1_d),
                  ((self.p2_u, self.p2_v), self.p2_d))
        return points

    def get_center_on_rgbd(self):
        points = ((self.pc_u, self.pc_v), self.pc_d)
        return points

    def _is_outside_frame(self) -> bool:
        """画面に入らない点はスキップ"""
        if self.p1_u < 0 or self.p1_u < 0 or self.h <= self.p1_v or self.w <= self.p1_u:
            return True
        if self.p2_u < 0 or self.p2_u < 0 or self.h <= self.p2_v or self.w <= self.p2_u:
            return True
        return False

    def _is_depth_missing(self) -> bool:
        """depthが取得できていない(値が0以下)の場合はスキップ"""
        if self.p1_d <= 0 or self.p2_d <= 0 or self.pc_d <= 0:
            return True
        return False

    def _is_in_mask(self, margin=1):
        """把持点がマスク内部に位置する場合はスキップ"""
        # measureDict=Trueのときpolygonの内側にptが存在する場合正の距離、輪郭上で０、外側で負の距離
        # TODO: marginをfinger_radiusから決定
        pt1_inner_dist = cv2.pointPolygonTest(
            self.contour, (self.p1_u, self.p1_v), measureDist=True)
        if pt1_inner_dist - margin > 0:
            return True
        pt2_inner_dist = cv2.pointPolygonTest(
            self.contour, (self.p2_u, self.p2_v), measureDist=True)
        if pt2_inner_dist - margin > 0:
            return True
        return False

    def _is_center_under_edges(self):
        """
        中心のdepthが把持点の深い方のdepthより深ければ（値が大きければ）スキップ
        * カメラと作業領域に傾きがある場合を考慮して２点のより深い(値が大きい)方を比較対象に
        * marginはdepth自体の誤差を考慮
        """
        # WARN: カメラと作業領域に傾きがある場合
        deeper_pt = max(self.p1_d, self.p2_d)
        return self.pc_d > deeper_pt


class CandidateEdgePoint:
    def __init__(self, edge: Tuple[float, float], center_d: Optional[int] = None, h: Optional[int] = None, w: Optional[int] = None, finger_radius: Optional[float] = None, depth: Optional[np.ndarray] = None, contour: Optional[np.ndarray] = None):
        # TODO: フィルタリングに関しては1点のdepthではなく半径finger_radius内の領域のdepthの平均をとるべきかも
        self.center_d = center_d
        self.finger_radius = finger_radius
        self.depth = depth
        self.contour = contour

        # ピクセル座標の整数化とフレームアウトの補正
        self.edge_u, is_over_range_u = self.clip_pixel_index(int(round(edge[0])), 0, w - 1)
        self.edge_v, is_over_range_v = self.clip_pixel_index(int(round(edge[1])), 0, h - 1)
        self.edge_d = depth[self.edge_v][self.edge_u] if type(depth) is np.ndarray else None
        is_over_range = is_over_range_u or is_over_range_v
        # フレームアウトしていたらその時点でinvalid
        self.is_valid = not is_over_range or self.validate()

    def clip_pixel_index(self, raw_value: int, min_value: int, max_value: int) -> Tuple[int, bool]:
        """画面に入らない点はスキップ"""
        if raw_value < min_value:
            value = min_value
            is_over_range = True
        elif raw_value > max_value:
            value = max_value
            is_over_range = True
        else:
            is_over_range = False
            value = raw_value

        return value, is_over_range

    def validate(self):
        is_invalid = False
        if self.edge_d:
            is_invalid = is_invalid or self._is_depth_missing()
        if type(self.contour) is np.ndarray:
            is_invalid = is_invalid or self._is_in_mask()
        if self.center_d is not None:
            is_invalid = is_invalid or self._is_center_under_edges()
        return not is_invalid

    def get_edge_on_rgb(self) -> Tuple[int, int]:
        return (self.edge_u, self.edge_v)

    def get_edge_on_rgbd(self) -> Tuple[Tuple[int, int], int]:
        return ((self.edge_u, self.edge_v), self.edge_d)


    def _is_depth_missing(self) -> bool:
        """depthが取得できていない(値が0以下)の場合はスキップ"""
        if self.edge_d <= 0:
            return True
        return False

    def _is_in_mask(self, margin=1):
        """把持点がマスク内部に位置する場合はスキップ"""
        # measureDict=Trueのときpolygonの内側にptが存在する場合正の距離、輪郭上で０、外側で負の距離
        # TODO: marginをfinger_radiusから決定
        # ptの要素がnumpy.intだとエラー
        edge_inner_dist = cv2.pointPolygonTest(self.contour, (self.edge_u, self.edge_v), measureDist=True)
        # edge_inner_dist = cv2.pointPolygonTest(
        #     self.contour, (self.edge_u, self.edge_v), measureDist=True)
        if edge_inner_dist - margin > 0:
            return True

        return False

    def _is_center_under_edges(self):
        """
        中心のdepthが把持点のdepthより深ければ（値が大きければ）スキップ
        * marginはdepth自体の誤差を考慮
        """
        # WARN: カメラと作業領域に傾きがある場合
        return self.center_d > self.edge_d


class Candidate:
    def __init__(self, edges: List[CandidateEdgePoint], angle: float):
        self.edges = edges
        self.angle = angle
        self.is_valid = np.all([edge.is_valid for edge in self.edges])

    def get_edges_on_rgb(self) -> List[Tuple[int, int]]:
        return [edge.get_edge_on_rgb() for edge in self.edges]

    def get_edges_on_rgbd(self) -> List[Tuple[Tuple[int, int], int]]:
        return [edge.get_edge_on_rgbd() for edge in self.edges]


class GraspDetector:
    def __init__(self, frame_size, finger_num, unit_angle=15, margin=3, finger_radius=1):
        self.h, self.w = frame_size
        self.margin = margin
        self.finger_radius = finger_radius

        self.finger_num = finger_num
        self.base_angle = 360 // self.finger_num  # ハンドの指間の角度 (等間隔前提)
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角
        self.candidate_num = self.base_angle // self.unit_angle

        base_cos, base_sin = np.cos(np.radians(self.base_angle)), np.sin(
            np.radians(self.base_angle))
        unit_cos, unit_sin = np.cos(np.radians(self.unit_angle)), np.sin(
            np.radians(self.unit_angle))
        self.base_rmat = np.array(
            [[base_cos, -base_sin], [base_sin, base_cos]])
        self.unit_rmat = np.array(
            [[unit_cos, -unit_sin], [unit_sin, unit_cos]])

    def detect(self, center: Tuple[int, int], radius: float, contour: Optional[np.ndarray] = None, depth: Optional[np.ndarray] = None, filter=True) -> List[Candidate]:
        base_finger_v = np.array([0, -1]) * radius  # 単位ベクトル x 半径
        candidates = []
        # 基準となる線分をbase_angleまでunit_angleずつ回転する (左回り)
        center_d = depth[center[1]][center[0]] if type(depth) is np.ndarray else None
        for i in range(self.candidate_num):
            edges = []
            finger_v = base_finger_v
            for _ in range(self.finger_num):
                edges.append(CandidateEdgePoint(edge=tuple(center + finger_v), center_d=center_d, depth=depth,
                                                h=self.h, w=self.w, contour=contour, finger_radius=self.finger_radius))
                finger_v = np.dot(finger_v, self.base_rmat)
            cnd = Candidate(edges, angle=self.unit_angle * i)

            base_finger_v = np.dot(base_finger_v, self.unit_rmat)

            if not filter or cnd.is_valid:
                candidates.append(cnd)

        return candidates
