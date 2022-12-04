import pickle

import matplotlib.pyplot as plt


def load_py2_pickle(path):
    with open(path, mode='rb') as f:
        # python2でつくったpickleはpython3ではエンコーディングエラーがでる
        # ref: https://qiita.com/f0o0o/items/4cdad7f3748741a3cf74
        # 自作msgが入ってくるとエラー出る
        data = pickle.load(f, encoding='latin1')

    return data


def imshow(img, show_axis=False, cmap=None):
    plt.imshow(img, cmap=cmap)
    if show_axis is False:
        plt.axis("off")
        # Not work: 一部余白が残る
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
