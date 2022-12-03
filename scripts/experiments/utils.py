import matplotlib.pyplot as plt


def imshow(img, show_axis=False):
    plt.imshow(img)
    if show_axis is False:
        plt.axis("off")
        # Not work: 一部余白が残る
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
