# %%
from timeit import timeit
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cProfile
# %%
img_shape = (100, 100)
base_img = np.zeros(img_shape, dtype=np.uint8)
center = (50, 50)
circle_img = base_img.copy()
cv2.circle(circle_img, center, 10, 255, thickness=-1)
plt.imshow(circle_img, cmap="gray")
# %%
cnt_img = base_img.copy()
contours, hierarchy = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(cnt_img, contours, -1, 255)
plt.imshow(cnt_img, cmap="gray")
# %%
line_img = base_img.copy()
line_pt1 = center
line_pt2 = (80, 20)
line_img = cv2.line(line_img, line_pt1, line_pt2, 255, 1)
plt.imshow(line_img, cmap="gray")

# %%
bitwise_img = base_img.copy()
cv2.bitwise_and(cnt_img, line_img, bitwise_img)
plt.imshow(bitwise_img, cmap="gray")
print([(x,y) for x,y in zip(*np.where(bitwise_img > 0))])

# %%
def compute_intersection_between_contour_and_line(img_shape, contour, line_pt1, line_pt2):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討x  
    """
    base_img = np.zeros(img_shape, dtype=np.uint8)
    cnt_img = base_img.copy()
    cv2.drawContours(cnt_img, [contour], -1, 255)
    line_img = base_img.copy()
    line_img = cv2.line(line_img, line_pt1, line_pt2, 255)
    bitwise_img = base_img.copy()
    cv2.bitwise_and(cnt_img, line_img, bitwise_img)
    intersection = [(x,y) for x,y in zip(*np.where(bitwise_img > 0))][0]
    return intersection


# %%
intersection = compute_intersection_between_contour_and_line(img_shape, contours[0], line_pt1, line_pt2)
print("intersection:", intersection)
# %%
# timeitやcProfileの引数にndarray形式が使えない、tupleやlistで渡すと今度は関数内部でエラー
# n = 1000
# cProfile.run(f"compute_intersection_between_contour_and_line({img_shape}, [[[50, 40]]], {line_pt1}, {line_pt2})")
# timeit(f"compute_intersection_between_contour_and_line({img_shape}, contour=[[[50, 40]]], line_pt1={line_pt1}, line_pt2={line_pt2})", number=n)

# ラップするとうまくいく
def wrapper():
    compute_intersection_between_contour_and_line((1000, 1000), contours[0], line_pt1, line_pt2)

cProfile.run("wrapper()")

# (100, 100)で0.01sec, (1000, 1000)で0.03sec
# %%
