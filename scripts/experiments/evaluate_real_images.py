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
                           extract_flont_mask_with_thresh, get_3c_gray,
                           refine_flont_mask, transform_ddi)
from utils import RealsenseBagHandler, imshow

# %%
# depthはuint8の３チャネルになってる
# img_path_list = sorted(glob(f"{SAMPLES_PATH}/real_images/color/*"))
# depth_path_list = sorted(glob(f"{SAMPLES_PATH}/real_images/depth/*"))

# img = cv2.imread(img_path_list[0])
# depth = cv2.cvtColor(cv2.imread(depth_path_list[0]), cv2.COLOR_BGR2GRAY)
path = glob(f"{SAMPLES_PATH}/realsense_viewer_bags/*")[0]
handler = RealsenseBagHandler(path, 640, 480, 30)


img, depth = handler.get_images()
print(img.dtype, depth.dtype)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

# %% depth欠損の確認
print(f"min: {depth.min()}, count: {len(depth[depth == depth.min()])}")
imshow(np.where(depth == depth.min(), 255, 0))
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
merged_mask = np.where(np.sum(res.masks, axis=0) > 0, 255, 0).astype("uint8")
# depthの欠損箇所があれば全体マスクから除外
valid_mask = np.where(merged_mask * depth > 0, 255, 0).astype("uint8")
ddi = transform_ddi(np.where(valid_mask > 0, depth, depth[depth > 0].mean()), 5)
opt_depth_th = compute_optimal_depth_thresh(depth, valid_mask, n=3)
print("opt_depth_th :", opt_depth_th)
# この段階でのflont_maskはコンテナ含んだり、インスタンスの見切れなどもそんざいする
raw_flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=3)
raw_flont_img = cv2.bitwise_and(img, img, mask=raw_flont_mask)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(merged_mask)
axes[1].imshow(ddi)
axes[2].imshow(raw_flont_img)
# %% depth filteringの結果とインスタンスセグメンテーションの結果のマージ (背景除去 & マスク拡大)
valid_masks = [res.masks[i] for i in range(res.num_instances) if res.scores[i] >= 0.75]
flont_mask = refine_flont_mask(raw_flont_mask, valid_masks, 0.3)
flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img)
axes[1].imshow(flont_mask)
axes[2].imshow(flont_img)


# %%
finger_num = 4
hand_radius_mm = 150
finger_radius_mm = 1
unit_angle = 15
frame_size = img.shape[:2]
fp = handler.fp
elements_th = 0
center_diff_th = 0
el_insertion_th = 0
el_contact_th = 0
el_bw_depth_th = 0
detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                         finger_radius_mm=finger_radius_mm,
                         unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                         elements_th=elements_th, center_diff_th=center_diff_th,
                         el_insertion_th=el_insertion_th, el_contact_th=el_contact_th,
                         el_bw_depth_th=el_bw_depth_th)
# %%
score_th = 0.8

gray_3c = get_3c_gray(img)
reversed_flont_mask = cv2.bitwise_not(flont_mask)
base_img = cv2.bitwise_and(img, img, mask=flont_mask) + \
    cv2.bitwise_and(gray_3c, gray_3c, mask=reversed_flont_mask)

print(res.num_instances)
cnd_img_1 = base_img.copy()
cnd_img_2 = base_img.copy()
for i in range(res.num_instances):
    label = str(res.labels[i])
    score = res.scores[i]
    center = res.centers[i]
    area = res.areas[i]
    mask = res.masks[i]
    contour = res.contours[i]

    if score < score_th:
        continue

    # try:
    candidates = detector.detect(center, depth, contour)

    for cnd in candidates:
        coef = 1 - cnd.total_score
        color = (255, 255 * coef, 255 * coef)
        if cnd.is_framein:
            cnd.draw(cnd_img_1, color)
            if cnd.is_valid:
                cnd.draw(cnd_img_2, color)
    print("success")
    cv2.circle(cnd_img_1, center, 3, (0, 0, 255), 1)
    cv2.drawContours(cnd_img_1, [contour], -1, (0, 100, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(cnd_img_2, center, 3, (0, 0, 255), 1)
    cv2.drawContours(cnd_img_2, [contour], -1, (0, 100, 255), 1, lineType=cv2.LINE_AA)
    # except Exception as e:
    #     print(i, e)

imshow(cnd_img_1)
imshow(cnd_img_2)
