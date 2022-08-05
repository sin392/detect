import cv2
import numpy as np


def get_optimal_hist_th(img, n):
    min_v, max_v = np.min(img[img > 0]), np.max(img)
    hist = cv2.calcHist(img, channels=[0], mask=None,
                        histSize=[65536], ranges=[0, 65536])
    h_list = np.array([])
    for i in range(min_v, max_v+1):
        t1 = np.sum(hist[i-n:i+n+1])
        t2 = np.sum(hist[i-n*2:i-n])
        t3 = np.sum(hist[i+n+1:i+n*2+1])
        res = t1 - t2 - t3
        h_list = np.append(h_list, res)
    sorted_h = np.argsort(h_list) + min_v
    return sorted_h


def extract_top_layer(img, depth, n):
    th = get_optimal_hist_th(depth, n)[0]
    flont_depth = np.where(depth < th, depth, 0)
    flont_depth_3d = np.array([flont_depth]*3).transpose(1, 2, 0)
    flont_img = img * (flont_depth_3d / 65535)
    flont_img = cv2.normalize(
        flont_img, dst=None, alpha=0,
        beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    return flont_img, flont_depth
