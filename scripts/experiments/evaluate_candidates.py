# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.const import SAMPLES_PATH
from modules.grasp import (compute_contact_point,
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
axes[0].imshow(candidate_img[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80])
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
imshow(candidate_img_2[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80])

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

imshow(candidate_img_3[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80])
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
cv2.circle(candidate_img_4, new_center, 3, (0, 0, 255), -1, cv2.LINE_AA)
imshow(candidate_img_4[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80])

# %%
