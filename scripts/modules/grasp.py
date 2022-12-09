from typing import List, Optional, Tuple

import cv2
import numpy as np
from modules.image import extract_depth_between_two_points

Px = float  # mmから変換した場合などは少数値になる
Mm = float
ImagePointUV = Tuple[int, int]  # [px, px, mm]
ImagePointUVD = Tuple[ImagePointUV, Mm]  # [px, px, mm]


class GraspCandidateElement:
    # ハンド情報、画像情報、局所的情報（ポイント、値）、しきい値
    def __init__(self, finger_radius: float, depth: np.ndarray, contour: np.ndarray, center: ImagePointUV, insertion_point: ImagePointUV):
        self.finger_radius = finger_radius

        self.center = center
        self.insertion_point = insertion_point
        self.center_d = depth[center[1], center[0]]
        self.insertion_point_d = depth[insertion_point[1], insertion_point[0]]

        # 詳細なスコアリングの前に明らかに不正な候補は弾く
        h, w = depth.shape[:2]
        self.is_valid_pre = self._precheck_validness(h, w)
        self.is_valid = self.is_valid_pre

        if self.is_valid_pre:
            pass

    def _precheck_validness(self, h: Px, w: Px) -> bool:
        is_valid_pre = \
            self._check_framein(h, w, self.center) and \
            self._check_framein(h, w, self.insertion_point) and \
            self._check_depth_existance() and \
            self._check_depth_difference()
        return is_valid_pre

    def _check_framein(self, h: Px, w: Px, pt: ImagePointUV) -> bool:
        return not (pt[0] < 0 or pt[1] < 0 or pt[0] >= w or pt[1] >= h)

    def _check_depth_existance(self) -> bool:
        return self.center_d != 0 and self.insertion_point_d != 0

    def _check_depth_difference(self) -> bool:
        return self.center_d <= self.insertion_point_d

    def get_insertion_point_uv(self) -> ImagePointUV:
        return self.insertion_point

    def get_insertion_point_uvd(self) -> ImagePointUVD:
        return (self.get_insertion_point_uv(), self.insertion_point_d)


class GraspCandidate:
    def __init__(self, finger_radius_px, angle, depth, contour, center, insertion_points):
        self.finger_radius_px = finger_radius_px
        self.angle = angle
        self.center = center
        self.elements = [GraspCandidateElement(finger_radius=finger_radius_px, depth=depth, contour=contour,
                                               center=center, insertion_point=insertion_point) for insertion_point in insertion_points]

        self.is_valid = True

    def get_insertion_points_uv(self) -> List[ImagePointUV]:
        return [el.get_insertion_point_uv() for el in self.elements]

    def get_insertion_points_uvd(self) -> List[ImagePointUVD]:
        return [el.get_insertion_point_uvd() for el in self.elements]


class GraspDetector:
    # TODO: hand_radiusの追加
    def __init__(self, finger_num, hand_radius_mm, finger_radius_mm, unit_angle, frame_size, fp):
        self.finger_num = finger_num
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角
        # NOTE: mm, pxの変換は深度、解像度、受光素子のサイズに依存するのでdetect時に変換
        self.hand_radius_mm = hand_radius_mm
        self.finger_radius_mm = finger_radius_mm

        self.h, self.w = frame_size
        self.fp = fp

        self.base_angle = 360 // self.finger_num  # ハンドの指間の角度 (等間隔前提)
        self.candidate_num = self.base_angle // self.unit_angle

        base_cos, base_sin = np.cos(np.radians(self.base_angle)), np.sin(
            np.radians(self.base_angle))
        unit_cos, unit_sin = np.cos(np.radians(self.unit_angle)), np.sin(
            np.radians(self.unit_angle))
        self.base_rmat = np.array(
            [[base_cos, -base_sin], [base_sin, base_cos]])
        self.unit_rmat = np.array(
            [[unit_cos, -unit_sin], [unit_sin, unit_cos]])

    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px

    # TODO: hand_radiusはインスタンス変数に
    def compute_insertion_points(self, center: ImagePointUV, base_finger_v: np.ndarray):
        finger_v = base_finger_v
        insertion_points = []
        for _ in range(self.finger_num):
            insertion_points.append(
                tuple(np.int0(np.round(center + finger_v))))
            finger_v = np.dot(finger_v, self.base_rmat)
        return insertion_points

    def detect(self, center: ImagePointUV, depth: Optional[np.ndarray] = None, contour: Optional[np.ndarray] = None, filter=True) -> List[GraspCandidate]:
        # 単位変換
        center_d = depth[center[1], center[0]]
        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(
            self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも
        base_finger_v = np.array([0, -1]) * hand_radius_px  # 単位ベクトル x ハンド半径
        candidates_list = []
        # 基準となる線分をbase_angleまでunit_angleずつ回転する (左回り)
        for i in range(self.candidate_num):
            finger_v = base_finger_v
            insertion_points = self.compute_insertion_points(center, finger_v)
            angle = self.unit_angle * i
            cnd = GraspCandidate(finger_radius_px=finger_radius_px, angle=angle,
                                 depth=depth, contour=contour, center=center, insertion_points=insertion_points)

            base_finger_v = np.dot(base_finger_v, self.unit_rmat)

            if not filter or cnd.is_valid:
                candidates_list.append(cnd)

        return candidates_list


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


# not in use now
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


def compute_bw_depth_profile(depth, contact_point, insertion_point):
    values = extract_depth_between_two_points(
        depth, contact_point, insertion_point)
    min_depth, max_depth = values.min(), values.max()
    # 欠損ピクセルの値は除外
    valid_values = values[values > 0]
    mean_depth = np.mean(valid_values) if len(valid_values) > 0 else 0
    return min_depth, max_depth, mean_depth


def compute_bw_depth_score(depth, contact_point, insertion_point, min_depth):
    _, max_depth, mean_depth = compute_bw_depth_profile(
        depth, contact_point, insertion_point)
    score = max(0, (mean_depth - min_depth)) / (max_depth - min_depth)
    return score


class _GraspCandidateElement:
    def __init__(self, depth, min_depth, max_depth, contour, center, insertion_point, finger_radius, insertion_score_thresh=0.5, contact_score_thresh=0.5, bw_depth_score_thresh=0):
        self.center = center
        self.insertion_point = insertion_point
        self.finger_radius = finger_radius

        self.contact_point = None
        self.contact_score = 0
        self.bw_depth_score = 0
        self.total_score = 0
        self.is_valid = False

        # TODO: ハンドの開き幅調整可能な場合 insertion point = contact pointとなるので、insertionのスコアはいらない
        # 挿入点の評価
        self.insertion_score = self._compute_point_score(
            depth, min_depth, max_depth)
        if self.insertion_score < insertion_score_thresh:
            return
        # 接触点の計算と評価
        self.intersection_point = self._compute_intersection_point(contour)
        self.contact_point = self._compute_contact_point(
            self.intersection_point)
        self.contact_score = self._compute_point_score(
            depth, min_depth, max_depth, self.contact_point)
        if self.contact_score < contact_score_thresh:
            return
        # 挿入点と接触点の間の障害物の評価
        self.bw_depth_score = self._compute_bw_depth_score(
            depth, self.contact_point)
        if self.bw_depth_score < bw_depth_score_thresh:
            return
        self.total_score = self._compute_total_score()
        # すべてのスコアが基準を満たしたときのみvalid判定
        self.is_valid = True

    def _compute_intersection_point(self, contour):
        # TODO: 個々のcropはまだ上位に引き上げられそう
        x, y, w, h = cv2.boundingRect(contour)
        upper_left_point = np.array((x, y))
        shifted_contour = contour - upper_left_point
        shifted_center, shifted_edge = [
            tuple(pt - upper_left_point) for pt in (self.center, self.insertion_point)]

        shifted_intersection = compute_intersection_between_contour_and_line(
            (h, w), shifted_contour, shifted_center, shifted_edge)
        intersection = tuple(shifted_intersection + upper_left_point)
        return intersection

    def _compute_contact_point(self, intersection_point):
        direction_v = np.array(self.insertion_point) - np.array(self.center)
        unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
        # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
        contact_point = np.int0(
            np.round(intersection_point + unit_direction_v * self.finger_radius))
        return contact_point

    def _compute_point_score(self, depth, min_depth, max_depth, point):
        # TODO: 引数のmin, maxはターゲットオブジェクト周辺の最小値・最大値
        _, _, mean_depth = compute_depth_profile_in_finger_area(
            depth, point[:: -1], self.finger_radius)
        score = max(0, (mean_depth - min_depth)) / \
            (max_depth - min_depth + 1e-6)
        return score

    def _compute_bw_depth_score(self, depth, contact_point):
        min_depth, max_depth, mean_depth = compute_bw_depth_profile(
            depth, contact_point, self.insertion_point)
        score = max(0, (mean_depth - min_depth)) / \
            (max_depth - min_depth + 1e-6)
        return score

    def _compute_total_score(self):
        # TODO: ip, cp間のdepthの評価 & 各項の重み付け
        return self.insertion_score * self.contact_score * self.bw_depth_score

    def get_points(self):
        return {"center": self.center, "intersection": self.intersection_point, "contact": self.contact_point, "insertion": self.insertion_point}

    def get_scores(self):
        return {"insertion": self.insertion_score, "contact": self.contact_score, "bw_depth": self.bw_depth_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True):
        cv2.line(img, self.center, self.insertion_point,
                 line_color, line_thickness, cv2.LINE_AA)
        if show_circle:
            cv2.circle(img, self.insertion_point, self.finger_radius,
                       (255, 0, 0), circle_thickness, cv2.LINE_AA)
            cv2.circle(img, self.contact_point, self.finger_radius,
                       (0, 255, 0), circle_thickness, cv2.LINE_AA)


class _GraspCandidate:
    def __init__(self, depth, min_depth, max_depth, contour, center, edges, finger_radius, hand_radius,
                 elements_score_thresh=0, center_diff_score_thresh=0, el_insertion_score_thresh=0.5, el_contact_score_thresh=0.5, el_bw_depth_score_thresh=0):
        self.center = center
        self.finger_radius = finger_radius
        self.hand_radius = hand_radius

        self.elements = [
            _GraspCandidateElement(depth, min_depth, max_depth, contour, center, edge,
                                   finger_radius, el_insertion_score_thresh,
                                   el_contact_score_thresh, el_bw_depth_score_thresh)
            for edge in edges]

        self.shifted_center = None
        self.elements_score = 0
        self.center_diff_score = 0
        self.total_score = 0
        self.is_valid = False

        self.elements_is_valid = self._merge_elements_validness()
        if self.elements_is_valid:
            # elementの組み合わせ評価
            self.elements_score = self._compute_elements_score()
            if self.elements_score < elements_score_thresh:
                return
            # contact pointsの中心とマスクの中心のズレの評価
            self.shifted_center = self._compute_contact_points_center()
            self.center_diff_score = self._compute_center_diff_score()
            if self.center_diff_score < center_diff_score_thresh:
                return
            # 各スコアの合算
            self.total_score = self._compute_total_score()
            # すべてのスコアが基準を満たしたときのみvalid判定
            self.is_valid = True

    def _merge_elements_validness(self):
        return np.all([el.is_valid for el in self.elements])

    def _compute_contact_points_center(self):
        contact_points = self.get_contact_points()
        return np.int0(np.round(np.mean(contact_points, axis=0)))

    def _compute_elements_score(self):
        element_scores = self.get_element_scores()
        # return np.prod(element_scores)
        # return np.mean(element_scores) * (np.min(element_scores) / np.max(element_scores))
        return (np.mean(element_scores) - np.min(element_scores)) / (np.max(element_scores) - np.min(element_scores))

    def _compute_center_diff_score(self):
        return 1. - (np.linalg.norm(np.array(self.center) - np.array(self.shifted_center), ord=2) / self.hand_radius)

    def _compute_total_score(self):
        return self.elements_score * self.center_diff_score

    def get_insertion_points(self):
        return tuple([el.insertion_point for el in self.elements])

    def get_contact_points(self):
        return tuple([el.contact_point for el in self.elements])

    def get_element_scores(self):
        return tuple([el.total_score for el in self.elements])

    def get_scores(self):
        return {"elements": self.elements_score, "center_diff": self.center_diff_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True):
        for el in self.elements:
            el.draw(img, line_color, line_thickness,
                    circle_thickness, show_circle)
