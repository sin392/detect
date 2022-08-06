import cv2
import numpy as np


def generate_candidates(center, bbox, mask, unit_angle, func):
    func = min if func == 'min' else max
    # bbox is (xmin, ymin, xmax, ymax)
    radius = func(np.linalg.norm(bbox[2]-bbox[0]),
                  np.linalg.norm(bbox[3]-bbox[1])) / 2
    v = np.array([0, -1])*radius  # 単位ベクトル x 半径
    cos, sin = np.cos(np.radians(unit_angle)), np.sin(np.radians(unit_angle))
    rmat = np.array([[cos, -sin], [sin, cos]])
    candidates = []
    for i in range(180//unit_angle):
        v = np.dot(v, rmat)  # 回転ベクトルの更新
        p1, p2 = center + v, center - v
        # TOFIX: ptにそのままp1, p2をわたすと何故かエラー
        p1, p2 = (int(p1[0].item()), int(p1[1].item())
                  ), (int(p2[0].item()), int(p2[1].item()))
        # 画面範囲を超える場合は候補から除外
        h, w = mask.shape
        if p1[0] < 0 or p1[1] < 0 or h <= p1[1] or w <= p1[0]:
            continue
        if p2[0] < 0 or p2[1] < 0 or h <= p2[1] or w <= p2[0]:
            continue
        candidates.append((p1, p2))


def generate_candidates_list(indexed_img, unit_angle=15, func='min'):
    cos, sin = np.cos(np.radians(unit_angle)), np.sin(np.radians(unit_angle))
    rmat = np.array([[cos, -sin], [sin, cos]])
    candidates_list = []  # インスタンスごとの把持候補領域を格納
    contours = []  # インスタンスごとの領域を格納
    boxes = []  # インスタンスごとの矩形領域を格納
    radiuses = []  # インスタンスごとの半径を格納
    centers = []
    for label in range(1, len(np.unique(indexed_img))):
        # 各インスタンスの重心とbboxの算出
        each_mask = np.where(indexed_img == label, 255, 0).astype('uint8')
        contours.extend(cv2.findContours(
            each_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        # NOTE: 重心算出もっと簡潔な方法ありそう
        mu = cv2.moments(contour)
        center = np.array(
            [int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])])
        centers.append(center)
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
        # bboxの短辺or長辺を半径とする
        func = min if func == 'min' else max
        # box is 左上, 右上, 右下, 左下
        radius = func(np.linalg.norm(box[0]-box[1]),
                      np.linalg.norm(box[1]-box[2])) / 2
        radius += 1  # margin
        radiuses.append(radius)
        v = np.array([0, -1])*radius  # 単位ベクトル x 半径
        candidates = []
        h, w = indexed_img.shape

        for i in range(180//unit_angle):
            v = np.dot(v, rmat)  # 回転ベクトルの更新
            p1, p2 = center + v, center - v
            # TOFIX: ptにそのままp1, p2をわたすと何故かエラー
            p1, p2 = (int(p1[0].item()), int(p1[1].item())
                      ), (int(p2[0].item()), int(p2[1].item()))
            # 画面範囲を超える場合は候補から除外
            if p1[0] < 0 or p1[1] < 0 or h <= p1[1] or w <= p1[0]:
                continue
            if p2[0] < 0 or p2[1] < 0 or h <= p2[1] or w <= p2[0]:
                continue
            # 自分よりも上位の物体に把持領域が重なった候補は除去
            # 背景は０とし、遠い物体から順に並んでいる
            # if label <= indexed_img[p1[::-1]] or label <= indexed_img[p2[::-1]]:
            #     continue
            # 始点と終点が共に外側にcontourの外側に存在する候補のみ保持
            p1_in_contour = cv2.pointPolygonTest(
                contour, p1, measureDist=False)
            p2_in_contour = cv2.pointPolygonTest(
                contour, p2, measureDist=False)
            if p1_in_contour == -1 and p2_in_contour == -1:
                # 除外よりは把持候補を長くするほうがいいのでは
                # (lineに沿ったcountourの淵抽出する？)
                candidates.append((p1, p2))
        # if len(candidates) > 0:
        candidates_list.append(candidates)
        contours.append(contour)
        boxes.append(box)

    return candidates_list, contours, boxes, radiuses, centers
