from typing import List, Tuple

import cv2
import numpy as np
import rospy


class ParallelCandidate:
    def __init__(self, p1, p2, pc, depth, h, w, contour, finger_radius=1):
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
        points = ((self.p1_u, self.p1_v, self.p1_d), (self.p2_u, self.p2_v, self.p2_d))
        return points

    def get_center_on_rgbd(self):
        points = (self.pc_u, self.pc_v, self.pc_d)
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
        pt1_inner_dist = cv2.pointPolygonTest(self.contour, (self.p1_u, self.p1_v), measureDist=True)
        if pt1_inner_dist - margin > 0:
            return True
        pt2_inner_dist = cv2.pointPolygonTest(self.contour, (self.p2_u, self.p2_v), measureDist=True)
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
    def __init__(self, frame_size: Tuple[int, int], unit_angle=15, margin=3):
        self.h, self.w = frame_size
        self.unit_angle = unit_angle
        self.margin = margin
        cos, sin = np.cos(np.radians(unit_angle)), np.sin(
            np.radians(unit_angle))
        self.rmat = np.array([[cos, -sin], [sin, cos]])

    def detect(self, center, radius, contour, depth, filter=True) -> List[ParallelCandidate]:
        v = np.array([0, -1]) * radius  # 単位ベクトル x (半径 + マージン)

        candidates = []
        # best_candidate = None
        for i in range(180 // self.unit_angle):
            v = np.dot(v, self.rmat)  # 回転ベクトルの更新
            cnd = ParallelCandidate(center + v, center - v, center, depth, self.h, self.w, contour)

            if filter and not cnd.is_valid:
                continue

            candidates.append(cnd)

        return candidates


def generate_candidates_list(indexed_img, unit_angle=15, margin=3, func='min'):
    cos, sin = np.cos(np.radians(unit_angle)), np.sin(np.radians(unit_angle))
    rmat = np.array([[cos, -sin], [sin, cos]])
    candidates_list = []  # インスタンスごとの把持候補領域を格納
    contours = []  # インスタンスごとの領域を格納
    boxes = []  # インスタンスごとの矩形領域を格納
    radiuses = []  # インスタンスごとの半径を格納
    centers = []
    for label in range(1, len(np.unique(indexed_img))):
        # 各インスタンスの重心とbboxの算出
        each_mask = np.where(indexed_img == label, 255, 0).astype('uint8')
        sub_contours, _ = cv2.findContours(
            each_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(sub_contours, key=lambda x: cv2.contourArea(x))
        contours.append(sub_contours)
        # NOTE: 重心算出もっと簡潔な方法ありそう
        mu = cv2.moments(contour)
        center = np.array(
            [int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])])
        centers.append(center)
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
        # bboxの短辺or長辺を半径とする
        func = min if func == 'min' else max
        # box is 左上, 右上, 右下, 左下
        radius = func(np.linalg.norm(box[0] - box[1]),
                      np.linalg.norm(box[1] - box[2])) / 2
        radius += margin  # margin
        radiuses.append(radius)
        v = np.array([0, -1]) * radius  # 単位ベクトル x 半径
        candidates = []
        h, w = indexed_img.shape

        for i in range(180 // unit_angle):
            v = np.dot(v, rmat)  # 回転ベクトルの更新
            p1, p2 = center + v, center - v
            # TOFIX: ptにそのままp1, p2をわたすと何故かエラー
            p1, p2 = (int(p1[0].item()), int(p1[1].item())
                      ), (int(p2[0].item()), int(p2[1].item()))
            # 画面範囲を超える場合は候補から除外
            if p1[0] < 0 or p1[1] < 0 or h <= p1[1] or w <= p1[0]:
                continue
            if p2[0] < 0 or p2[1] < 0 or h <= p2[1] or w <= p2[0]:
                continue
            # 自分よりも上位の物体に把持領域が重なった候補は除去
            # 背景は０とし、遠い物体から順に並んでいる
            # if label <= indexed_img[p1[::-1]] or label <= indexed_img[p2[::-1]]:
            #     continue
            # 始点と終点が共に外側にcontourの外側に存在する候補のみ保持
            p1_in_contour = cv2.pointPolygonTest(
                contour, p1, measureDist=False)
            p2_in_contour = cv2.pointPolygonTest(
                contour, p2, measureDist=False)
            if p1_in_contour == -1 and p2_in_contour == -1:
                # 除外よりは把持候補を長くするほうがいいのでは
                # (lineに沿ったcountourの淵抽出する？)
                candidates.append((p1, p2))
        # if len(candidates) > 0:
        candidates_list.append(candidates)
        contours.append(contour)
        boxes.append(box)

    return candidates_list, contours, boxes, radiuses, centers
