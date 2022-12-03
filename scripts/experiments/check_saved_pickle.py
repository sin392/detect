# %%
import os
import pickle
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
masks = []
contours = []
centers = []
candidates_list = []
for obj in objects:
    masks.append(obj["mask"])
    contours.append(obj["contour"])
    centers.append(obj["center"])
    candidates_list.append(obj["candidates_list"])

# %%
merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0)
plt.imshow(merged_mask)
# %%
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (255, 255, 0), -1)
alpha = 0.1
overlay_img = cv2.addWeighted(img, 1 - alpha, contour_img, alpha, 0)
cv2.drawContours(overlay_img, contours, -1, (255, 255, 0), 2)
plt.imshow(overlay_img)
# %%
candidate_img = overlay_img.copy()
target_index = 0

target_candidates = candidates_list[target_index]
print(np.array(candidates_list).shape)
target_center = centers[target_index]
for points_list in target_candidates:
    for points in points_list:
        for edge in points:
            # print(target_center, edge)
            cv2.line(candidate_img, target_center, np.int0(edge), (255, 0, 0), 1)
    break

for i in range(len(centers)):
    center = centers[i]
    cv2.circle(candidate_img, center, 3, (255, 0, 0), -1)
    cv2.putText(candidate_img, str(i), (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)


plt.imshow(candidate_img)
# %%
