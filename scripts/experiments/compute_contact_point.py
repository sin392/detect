# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cProfile
# %%
img_shape = (1000, 1000)
base_img = np.zeros(img_shape, dtype=np.uint8)
center = (50, 80)
circle_img = base_img.copy()
cv2.circle(circle_img, center, 10, 255, thickness=-1)
plt.imshow(circle_img, cmap="gray")
# %%
contours, hierarchy = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
x,y,w,h = cv2.boundingRect(contours[0])
upper_left_point = np.array((x,y))
print(x,y,w,h)
print(contours)
shifted_contours = [contour - upper_left_point for contour in contours]
cropped_base_img = base_img[y:y+h, x:x+w]
cnt_img = cropped_base_img.copy()
cv2.drawContours(cnt_img, shifted_contours, -1, 255)
plt.imshow(cnt_img, cmap="gray")
# %%
line_img = cropped_base_img.copy()
line_pt1 = center
line_pt2 = (80, 20)
shifted_line_pt1 = center - upper_left_point
shifted_line_pt2 = (80, 20) - upper_left_point
line_img = cv2.line(line_img, tuple(shifted_line_pt1), tuple(shifted_line_pt2), 255, 1)
plt.imshow(line_img, cmap="gray")

# %%
bitwise_img = cropped_base_img.copy()
cv2.bitwise_and(cnt_img, line_img, bitwise_img)
plt.imshow(bitwise_img, cmap="gray")
intersection = [(x,y) for x,y in zip(*np.where(bitwise_img > 0))][0]
original_intersection = tuple(intersection + upper_left_point)
print("intersection:", intersection)
print("original intersection:", original_intersection)

# %%
def compute_intersection_between_contour_and_line(contour, line_pt1, line_pt2):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
    """
    # TODO: クロップはマスクで共通なので外に出したい、クラスとしてまとめてもいいかも
    x,y,_,_ = cv2.boundingRect(contours[0])
    upper_left_point = np.array((x,y))
    base_img = np.zeros(upper_left_point, dtype=np.uint8)

    cnt_img = base_img.copy()
    shifted_contour = contour - upper_left_point
    cv2.drawContours(cnt_img, [shifted_contour], -1, 255)
    line_img = base_img.copy()
    shifted_line_pt1, shifted_line_pt2 = [tuple(pt - upper_left_point) for pt in (line_pt1, line_pt2)]
    line_img = cv2.line(line_img, shifted_line_pt1, shifted_line_pt2, 255)
    bitwise_img = base_img.copy()
    cv2.bitwise_and(cnt_img, line_img, bitwise_img)
    intersection = [(x,y) for x,y in zip(*np.where(bitwise_img > 0))][0]
    original_intersection = tuple(intersection + upper_left_point)
    return original_intersection


# %%
intersection = compute_intersection_between_contour_and_line(contours[0], line_pt1, line_pt2)
print("intersection:", intersection)
# %%
# timeitやcProfileの引数にndarray形式が使えない、tupleやlistで渡すと今度は関数内部でエラー
# n = 1000
# cProfile.run(f"compute_intersection_between_contour_and_line({img_shape}, [[[50, 40]]], {line_pt1}, {line_pt2})")
# timeit(f"compute_intersection_between_contour_and_line({img_shape}, contour=[[[50, 40]]], line_pt1={line_pt1}, line_pt2={line_pt2})", number=n)

# ラップするとうまくいく
def wrapper():
    compute_intersection_between_contour_and_line(contours[0], line_pt1, line_pt2)

cProfile.run("wrapper()")

# (100, 100)で0.01sec, (1000, 1000)で0.03sec