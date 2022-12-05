# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.const import SAMPLES_PATH
from modules.grasp import (compute_bw_depth_profile, compute_contact_point,
                           compute_depth_profile_in_finger_area,
                           evaluate_single_insertion_point)
from utils import imshow, load_py2_pickle

# %%
path_list = sorted(glob(f"{SAMPLES_PATH}/saved_data/*"))
path = path_list[0]

data = load_py2_pickle(path)

img = data["img"]
depth = data["depth"]
objects = data["objects"]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

# %%


def crop(img, center, width):
    half_width = width // 2
    return img[center[1] - half_width:center[1] + half_width, center[0] - half_width:center[0] + half_width]

# %%


def compute_min_max_depth(depth, mask=None):
    if mask is None:
        return depth.min(), depth.max()  # global min max
    else:
        values = depth[mask > 0]
        return values.min(), values.max()  # objects min max


finger_radius = 10
merged_mask = np.where(np.sum([obj["mask"] for obj in objects], axis=0) > 0, 255, 0).astype("uint8")
objects_min_depth, objects_max_depth = compute_min_max_depth(depth, merged_mask)
print("objects range:", objects_min_depth, objects_max_depth)
# %%
target_index = 3
candidate_img = img.copy()
insertion_points_info = []
for i, obj in enumerate(objects):
    if i != target_index:
        continue
    candidates = obj["candidates"]
    contour = obj["contour"]
    mask = obj["mask"]
    center = obj["center"]

    instance_min_depth = depth[mask > 0].min()

    for j, points in enumerate(candidates):
        for edge in points:
            score = evaluate_single_insertion_point(depth, edge[::-1], finger_radius, instance_min_depth, objects_max_depth)
            insertion_points_info.append({"edge": edge, "score": score, "candidate_idx": j})
            cv2.line(candidate_img, center, np.int0(edge), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(candidate_img, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)

print(len(insertion_points_info))
fig, axes = plt.subplots(1, 2)
axes[0].imshow(crop(candidate_img, center, 160))
axes[1].hist([obj["score"] for obj in insertion_points_info])

# %%
# ip_scoreによるスレッショルド
ip_score_thresh = 0.8
finger_num = 4
candidate_img_2 = img.copy()
valid_candidates = []
for i in range(0, len(insertion_points_info) + 1, finger_num):
    ip_infos = insertion_points_info[i:i + finger_num]
    invalid_flg = False
    valid_edges = []
    for ip_info in ip_infos:
        edge = ip_info["edge"]
        score = ip_info["score"]
        if score < ip_score_thresh:
            invalid_flg = True
        else:
            valid_edges.append(edge)
    for edge in valid_edges:
        color = (50, 50, 100) if invalid_flg else (255, 100, 0)
        cv2.line(candidate_img_2, center, np.int0(edge), color, 1, cv2.LINE_AA)
        cv2.circle(candidate_img_2, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
    if not invalid_flg and len(valid_edges):
        valid_candidates.append(valid_edges)

print(valid_candidates)
imshow(crop(candidate_img_2, center, 160))

# %%
# 生き残ったinsertion pointからcontact pointを計算
candidate_img_3 = img.copy()

candidate_ip_scores = []
candidate_cp_scores = []
for points in valid_candidates:
    tmp_ip_scores = []
    tmp_cp_scores = []
    for edge in points:
        contact_point = compute_contact_point(contour, center, edge, finger_radius)
        tmp_ip_scores.append(evaluate_single_insertion_point(depth, edge[::-1], finger_radius, instance_min_depth, objects_max_depth))
        tmp_cp_scores.append(evaluate_single_insertion_point(depth, contact_point[::-1], finger_radius, instance_min_depth, objects_max_depth))
        cv2.line(candidate_img_3, center, np.int0(edge), (255, 100, 0), 1, cv2.LINE_AA)
        cv2.circle(candidate_img_3, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(candidate_img_3, contact_point, finger_radius, (0, 255, 0), 1, cv2.LINE_AA)
    candidate_ip_scores.append(tmp_ip_scores)
    candidate_cp_scores.append(tmp_cp_scores)

print("ip scores", candidate_ip_scores)
print("cp scores", candidate_cp_scores)
candidate_edge_scores = [list(a * b) for a, b in zip(np.array(candidate_ip_scores), np.array(candidate_cp_scores))]
print("ip x cp scores", candidate_edge_scores)
# TODO: contact pointの中心をもとめて、マスクの重心との差をcandidateのスコアに盛り込む
candidate_scores = [np.prod(scores) for scores in candidate_edge_scores]
print("candidate scores", candidate_scores)

imshow(crop(candidate_img_3, center, 160))
# %%
candidate_img_4 = img.copy()
best_index = np.argmax(candidate_scores)
best_candidate = valid_candidates[best_index]
final_contact_points = []
for edge in best_candidate:
    contact_point = compute_contact_point(contour, center, edge, finger_radius)
    final_contact_points.append(contact_point)
    cv2.line(candidate_img_4, center, np.int0(edge), (255, 100, 0), 2, cv2.LINE_AA)
    cv2.circle(candidate_img_4, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(candidate_img_4, contact_point, finger_radius, (0, 255, 0), 1, cv2.LINE_AA)
hand_radius = np.linalg.norm(np.array(center) - np.array(edge), ord=2)  # const
print(hand_radius)
new_center = np.int0(np.round(np.mean(final_contact_points, axis=0)))
center_diff_score = 1 - (np.linalg.norm(np.array(center) - np.array(new_center), ord=2) / hand_radius)
print(center, new_center, center_diff_score)
cv2.circle(candidate_img_4, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
cv2.circle(candidate_img_4, new_center, 3, (0, 255, 0), -1, cv2.LINE_AA)
imshow(crop(candidate_img_4, center, 160))

# %%


class GraspCandidateElement:
    def __init__(self, depth, min_depth, max_depth, contour, center, edge, finger_radius, score_thresh=0):
        self.center = center
        self.insertion_point = edge
        # TODO: ハンドの開き幅調整可能な場合 insertion point = contact pointとなるので、insertionのスコアはいらない
        self.insertion_score = self.compute_point_score(self.insertion_point, depth, min_depth, max_depth, finger_radius)

        if self.insertion_score >= score_thresh:
            self.contact_point = self.compute_contact_point(contour, finger_radius)
            self.contact_score = self.compute_point_score(self.contact_point, depth, min_depth, max_depth, finger_radius)
            self.bw_depth_score = self.compute_bw_depth_score(depth, min_depth)
            self.total_score = self.compute_total_score()
            # self.is_valid = self.bw_mean_depth - min_depth > 20
            self.is_valid = True
        else:
            # Noneとか入れるとあとで面倒になったりしないか
            self.contact_point = None
            self.contact_score = None
            # self.bw_mean_depth = None
            self.total_score = 0.
            self.is_valid = False

    def compute_contact_point(self, contour, finger_radius):
        contact_point = tuple(compute_contact_point(contour, self.center, self.insertion_point, finger_radius))
        return contact_point

    def compute_point_score(self, point, depth, min_depth, max_depth, finger_radius):
        # score =  evaluate_single_insertion_point(depth, point[::-1], finger_radius, min_depth, max_depth)
        _, _, mean_depth = compute_depth_profile_in_finger_area(depth, point[::-1], finger_radius)
        score = ((mean_depth - min_depth) / (max_depth - min_depth + 1e-6)) ** 2
        return score

    def compute_bw_depth_score(self, depth, min_depth):
        # score = compute_bw_depth_score(depth, self.contact_point, self.insertion_point, min_depth)
        _, max_depth, mean_depth = compute_bw_depth_profile(depth, self.contact_point, self.insertion_point)
        # self.bw_mean_depth = mean_depth
        # score = max(0, (mean_depth - min_depth)) / (max_depth - min_depth)
        score = (max(0, (mean_depth - min_depth)) / (max_depth - min_depth)) ** 2
        return score

    def compute_total_score(self):
        # TODO: ip, cp間のdepthの評価 & 各項の重み付け
        return self.insertion_score * self.contact_score * self.bw_depth_score

    def check_invalidness(self, thresh):
        return self.total_score < thresh

    def get_points(self):
        return {"center": self.center, "contact": self.contact_point, "insertion": self.insertion_point}

    def get_scores(self):
        return {"insertion": self.insertion_score, "contact": self.contact_score, "bw_depth": self.bw_depth_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True):
        cv2.line(img, self.center, self.insertion_point, line_color, line_thickness, cv2.LINE_AA)
        if show_circle:
            cv2.circle(img, self.insertion_point, finger_radius, (255, 0, 0), circle_thickness, cv2.LINE_AA)
            cv2.circle(img, self.contact_point, finger_radius, (0, 255, 0), circle_thickness, cv2.LINE_AA)

        return img


gce = GraspCandidateElement(depth, instance_min_depth, objects_max_depth, contour, center, edge, finger_radius)
print(gce.get_points())
print(gce.get_scores())
print("total score:", gce.total_score)

test_img = img.copy()
gce.draw(test_img, (255, 100, 0))

imshow(crop(test_img, center, 160))
# %%


class GraspCandidate:
    def __init__(self, depth, min_depth, max_depth, contour, center, edges, finger_radius, hand_radius, candidate_score_thresh=0, element_score_thresh=0):
        self.center = center
        self.elements = [GraspCandidateElement(depth, min_depth, max_depth, contour, center, edge, finger_radius, element_score_thresh) for edge in edges]
        self.elements_is_valid = self.merge_elements_validness()

        if self.elements_is_valid:
            self.shifted_center = self.compute_contact_points_center()
            self.elements_score = self.compute_elements_score()
            self.center_diff_score = self.compute_center_diff_score(hand_radius)
            self.total_score = self.compute_total_score()
        else:
            self.shifted_center = None
            self.elements_score = None
            self.center_diff_score = None
            self.total_score = 0.

        self.is_valid = self.elements_is_valid or \
            self.total_score >= candidate_score_thresh

    def merge_elements_validness(self):
        return np.all([el.is_valid for el in self.elements])

    def compute_contact_points_center(self):
        contact_points = self.get_contact_points()
        return np.int0(np.round(np.mean(contact_points, axis=0)))

    def compute_elements_score(self):
        element_scores = self.get_element_scores()
        # return np.prod(element_scores)
        # return np.mean(element_scores) * (np.min(element_scores) / np.max(element_scores))
        return (np.mean(element_scores) - np.min(element_scores)) / (np.max(element_scores) - np.min(element_scores))

    def compute_center_diff_score(self, hand_radius):
        return 1. - (np.linalg.norm(np.array(self.center) - np.array(self.shifted_center), ord=2) / hand_radius)

    def compute_total_score(self):
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
            el.draw(img, line_color, line_thickness, circle_thickness, show_circle)
        return img


gc = GraspCandidate(depth, instance_min_depth, objects_max_depth, contour, center, points, finger_radius, hand_radius, 0, 0.5)
print(gc.get_insertion_points())
print(gc.get_contact_points())
print(gc.get_scores())

test_img = img.copy()
gc.draw(test_img, (255, 100, 0), 2, 0)
imshow(crop(test_img, gc.center, 160))

# %%
element_score_thresh = 0.7
candidate_score_thresh = 0.1
candidate_img = img.copy()
for i, obj in enumerate(objects):
    candidates = obj["candidates"]
    mask = obj["mask"]
    contour = obj["contour"]
    center = obj["center"]
    instance_min_depth = depth[mask > 0].min()

    gc_list = []
    best_score = 0
    best_index = 0
    for j, points in enumerate(candidates):
        gc = GraspCandidate(depth, instance_min_depth, objects_max_depth, contour, center, points, finger_radius, hand_radius, candidate_score_thresh, element_score_thresh)
        if gc.is_valid:
            print(i + j, gc.total_score, instance_min_depth, [el.insertion_score for el in gc.elements])
            # validなcandidateの多さはインスタンスの優先順位決定に使えそう
            gc_list.append(gc)
            if gc.total_score > best_score:
                best_score = gc.total_score
                best_index = j
            coef = ((1 - gc.total_score) ** 2)
            color = (255, 255 * coef, 255 * coef)
            min_insertion_score = np.min([el.insertion_score for el in gc.elements])
            # if min_insertion_score <= 0.9:
            #     color = (0, 0, 255)
            gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)

    cv2.circle(candidate_img, center, 3, (0, 0, 255), -1, cv2.LINE_AA)

imshow(candidate_img)

# %%
