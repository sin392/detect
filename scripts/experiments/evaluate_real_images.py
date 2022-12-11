# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, OUTPUTS_PATH, SAMPLES_PATH
from modules.grasp import GraspDetector
from modules.image import (compute_optimal_depth_thresh,
                           extract_flont_mask_with_thresh, transform_ddi)
from utils import imshow

# %%
# depthはuint8の３チャネルになってる
img_path_list = sorted(glob(f"{SAMPLES_PATH}/real_images/color/*"))
depth_path_list = sorted(glob(f"{SAMPLES_PATH}/real_images/depth/*"))

img = cv2.imread(img_path_list[0])
depth = cv2.cvtColor(cv2.imread(depth_path_list[0]), cv2.COLOR_BGR2GRAY)

print(img.dtype, depth.dtype)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

# %%
config_path = f"{CONFIGS_PATH}/config.yaml"
weight_path = f"{OUTPUTS_PATH}/2022_10_16_08_01/model_final.pth"
device = "cuda:0"

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = weight_path
cfg.MODEL.DEVICE = device

predictor = Predictor(cfg)

# %%
res = predictor.predict(img)
seg = res.draw_instances(img[:, :, ::-1])
imshow(seg)

# %%
merged_mask = np.where(np.sum(res.mask_array, axis=0) > 0, 255, 0).astype("uint8")
ddi = transform_ddi(depth, 5)
opt_depth_th = compute_optimal_depth_thresh(depth, merged_mask, n=3)
print("opt_depth_th :", opt_depth_th)
flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=3)
flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
imshow(flont_img)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(merged_mask)
axes[1].imshow(ddi)
axes[2].imshow(flont_img)
# %%
finger_num = 4
hand_radius_mm = 150
finger_radius_mm = 1
unit_angle = 15
frame_size = img.shape[:2]
fp = 55  # 適当
elements_th = 0
center_diff_th = 0
el_insertion_th = 0.1
el_contact_th = 0.1
el_bw_depth_th = 0.1
detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                         finger_radius_mm=finger_radius_mm,
                         unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                         elements_th=elements_th, center_diff_th=center_diff_th,
                         el_insertion_th=el_insertion_th, el_contact_th=el_contact_th,
                         el_bw_depth_th=el_bw_depth_th)


# %%
score_th = 0.8

print(res.num_instances)
cnd_img = flont_img.copy()
for i in range(res.num_instances):
    label = str(res.labels[i])
    score = res.scores[i]
    center = res.centers[i]
    area = res.areas[i]
    mask = res.mask_array[i]
    contour = res.contours[i]

    if score < score_th:
        continue

    # try:
    candidates = detector.detect(center, depth, contour)

    for cnd in candidates:
        coef = 1 - cnd.total_score
        color = (255, 255 * coef, 255 * coef)
        if cnd.is_framein:
            cnd.draw(cnd_img, color)
    print("success")
    cv2.circle(cnd_img, center, 3, (255, 0, 0), 1)
    cv2.drawContours(cnd_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)
    # except Exception as e:
    #     print(i, e)

imshow(cnd_img)

# %%
