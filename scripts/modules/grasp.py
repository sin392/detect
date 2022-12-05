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
        self.edge_u, is_over_range_u = self.clip_pixel_index(
            int(round(edge[0])), 0, w - 1)
        self.edge_v, is_over_range_v = self.clip_pixel_index(
            int(round(edge[1])), 0, h - 1)
        self.edge_d = depth[self.edge_v][self.edge_u] if type(
            depth) is np.ndarray else None
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
        edge_inner_dist = cv2.pointPolygonTest(
            self.contour, (self.edge_u, self.edge_v), measureDist=True)
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
    def __init__(self, edges: List[CandidateEdgePoint], angle: float, is_valid: bool):
        self.edges = edges
        self.angle = angle
        self.is_valid = is_valid

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
        center_d = depth[center[1]][center[0]] if type(
            depth) is np.ndarray else None
        for i in range(self.candidate_num):
            edges = []
            finger_v = base_finger_v
            is_invalid = False
            for _ in range(self.finger_num):
                cdp = CandidateEdgePoint(edge=tuple(center + finger_v), center_d=center_d, depth=depth,
                                         h=self.h, w=self.w, contour=contour, finger_radius=self.finger_radius)
                is_invalid = is_invalid or not cdp.is_valid
                edges.append(cdp)
                finger_v = np.dot(finger_v, self.base_rmat)
            cnd = Candidate(edges, angle=self.unit_angle *
                            i, is_valid=(not is_invalid))

            base_finger_v = np.dot(base_finger_v, self.unit_rmat)

            if not filter or cnd.is_valid:
                candidates.append(cnd)

        return candidates


def compute_depth_profile_in_finger_area(depth, pt_xy, radius):
    x_slice = slice(pt_xy[0] - radius, pt_xy[0] + radius + 1)
    y_slice = slice(pt_xy[1] - radius, pt_xy[1] + radius + 1)
    cropped_depth = depth[x_slice, y_slice]
    finger_mask = np.zeros_like(cropped_depth, dtype=np.uint8)
    cv2.circle(
        finger_mask, (cropped_depth.shape[0] // 2, cropped_depth.shape[1] // 2), radius, 255, -1)
    depth_values_in_mask = cropped_depth[finger_mask == 255]
    return int(np.min(depth_values_in_mask)), int(np.max(depth_values_in_mask)), int(np.mean(depth_values_in_mask))


def insertion_point_score(min_depth, max_depth, mean_depth):
    return ((mean_depth - min_depth) / (max_depth - min_depth + 1e-6)) ** 2


def evaluate_single_insertion_point(depth, pt_xy, radius, min_depth, max_depth):
    _, _, mean_depth = compute_depth_profile_in_finger_area(
        depth, pt_xy, radius)
    score = insertion_point_score(min_depth, max_depth, mean_depth)
    return score


def evaluate_insertion_points(depth, candidates, radius, min_depth, max_depth):
    scores = []
    for points in candidates:
        for u, v in points:
            score = evaluate_single_insertion_point(
                depth, (v, u), radius, min_depth, max_depth)
            scores.append(score)

    return scores


def evaluate_single_insertion_points_set(insertion_point_scores):
    """ 同じのcandidateに所属するinsertion_pointのスコアの積 """
    return np.prod(insertion_point_scores)


def evaluate_insertion_points_set(depth, candidates, radius, min_depth, max_depth):
    scores = evaluate_insertion_points(
        depth, candidates, radius, min_depth, max_depth)
    finger_num = len(candidates[0])
    candidates_scores = [evaluate_single_insertion_points_set(
        scores[i:i + finger_num]) for i in range(0, len(scores), finger_num)]

    return candidates_scores


def compute_intersection_between_contour_and_line(img_shape, contour, line_pt1_xy, line_pt2_xy):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
    """
    blank_img = np.zeros(img_shape)
    # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
    cnt_img = blank_img.copy()
    cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)
    # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
    line_img = blank_img.copy()
    # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
    line_img = cv2.line(line_img, line_pt1_xy, line_pt2_xy,
                        255, 2, lineType=cv2.LINE_AA)
    # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
    bitwise_img = blank_img.copy()
    cv2.bitwise_and(cnt_img, line_img, bitwise_img)

    intersections = [(w, h)
                     for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
    mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
    return mean_intersection


def compute_contact_point(contour, center, edge, finger_radius):
    x, y, w, h = cv2.boundingRect(contour)
    upper_left_point = np.array((x, y))
    shifted_contour = contour - upper_left_point
    shifted_center, shifted_edge = [
        tuple(pt - upper_left_point) for pt in (center, edge)]

    shifted_intersection = compute_intersection_between_contour_and_line(
        (h, w), shifted_contour, shifted_center, shifted_edge)
    intersection = tuple(shifted_intersection + upper_left_point)

    direction_v = np.array(edge) - np.array(center)
    unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
    # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
    contact_point = np.int0(
        np.round(intersection + unit_direction_v * finger_radius))

    return contact_point
