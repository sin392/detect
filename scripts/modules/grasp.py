from typing import List, Tuple

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


class ParallelGraspDetector:
    def __init__(self, frame_size: Tuple[int, int], unit_angle=15, margin=3, finger_radius=1):
        self.h, self.w = frame_size
        self.unit_angle = unit_angle
        self.margin = margin
        self.finger_radius = finger_radius
        cos, sin = np.cos(np.radians(unit_angle)), np.sin(
            np.radians(unit_angle))
        self.rmat = np.array([[cos, -sin], [sin, cos]])

    def detect(self, center, radius, contour, depth, filter=True) -> List[ParallelCandidate]:
        v = np.array([0, -1]) * radius  # 単位ベクトル x (半径 + マージン)

        candidates = []
        # best_candidate = None
        for i in range(180 // self.unit_angle):
            v = np.dot(v, self.rmat)  # 回転ベクトルの更新
            cnd = ParallelCandidate(
                center + v, center - v, center, depth, self.h, self.w, contour, self.finger_radius)

            if filter and not cnd.is_valid:
                continue

            candidates.append(cnd)

        return candidates
