# %%
import os
import pickle
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from modules.const import SAMPLES_PATH

# %%
path_list = sorted(glob(f"{SAMPLES_PATH}/saved_data/*"))
print(os.getcwd())
print(path_list)
# %%
path = path_list[0]
print(path)
with open(path, mode='rb') as f:
    # python2でつくったpickleはpython3ではエンコーディングエラーがでる
    # ref: https://qiita.com/f0o0o/items/4cdad7f3748741a3cf74
    # 自作msgが入ってくるとエラー出る
    data = pickle.load(f, encoding='latin1')

    img = data["img"]
    depth = data["depth"]
    objects = data["objects"]

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(depth, cmap="binary")
# %%
normalized_depth = (depth.copy() - depth.min()) / (depth.max() - depth.min())

grad_img = ((1 - normalized_depth[:, :, np.newaxis]) * 255).astype("uint8")
heatmap_img = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)

base_img = cv2.addWeighted(img, 1, heatmap_img, 0.2, 0)
plt.imshow(base_img)

# %%
masks = []
contours = []
centers = []
candidates_list = []
for obj in objects:
    masks.append(obj["mask"])
    contours.append(obj["contour"])
    centers.append(obj["center"])
    candidates_list.append(obj["candidates"])

# %%
merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0)
plt.imshow(merged_mask)
# %%
contour_img = base_img.copy()
cv2.drawContours(contour_img, contours, -1, (255, 255, 0), -1)
alpha = 0.1
overlay_img = cv2.addWeighted(base_img, 1 - alpha, contour_img, alpha, 0)
cv2.drawContours(overlay_img, contours, -1, (255, 255, 0), 2)
plt.imshow(overlay_img)
# %%
candidate_img = overlay_img.copy()
target_index = 3
target_candidate_index = 1

target_candidates = candidates_list[target_index]
target_center = centers[target_index]
for i, points in enumerate(target_candidates):
    is_target_candidate = target_candidate_index == i
    color = (240, 160, 80) if is_target_candidate else (0, 0, 255)
    thickness = 2 if is_target_candidate else 1
    for edge in points:
        cv2.line(candidate_img, target_center, np.int0(edge), color, thickness, cv2.LINE_AA)

for i in range(len(centers)):
    center = centers[i]
    cv2.circle(candidate_img, center, 3, (255, 0, 0), -1)
    cv2.putText(candidate_img, str(i), (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)


plt.imshow(candidate_img)
# %%
# about the third instance
flatten_points = sum(target_candidates, [])
print(len(flatten_points))
print(flatten_points)

depth_list = [depth[uv[::-1]] for uv in flatten_points]
print(depth_list)
fig, ax = plt.subplots()
ax.set_xlim(0, len(flatten_points) - 1)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.plot(depth_list)
# %%
# insertion pointのdepthによる判定
deep_points = [flatten_points[i] for i in range(len(flatten_points)) if depth_list[i] > 1060]
print(len(deep_points))

candidate_img_2 = candidate_img.copy()
for pt in deep_points:
    cv2.circle(candidate_img_2, pt, 3, (0, 255, 0), -1)
    cv2.circle(candidate_img_2, pt, 10, (0, 100, 100), 1, cv2.LINE_AA)
plt.imshow(candidate_img_2)

# %%
