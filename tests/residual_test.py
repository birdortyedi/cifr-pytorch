import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

img_ids = ["37_7", "88_9"]
diff_names = ["CIFR", "IFRNet", "GcGAN", "AngularGAN"]
residuality = 80


def extract_error_map(img_id, residuality=127.5):
    print(img_id)
    cifr_img = cv2.imread("../../Downloads/fig3/{}/{}_cifr.png".format(img_id, img_id)).astype(np.float)
    ifrnet_img = cv2.imread("../../Downloads/fig3/{}/{}_ifrnet.png".format(img_id, img_id)).astype(np.float)
    angular_img = cv2.imread("../../Downloads/fig3/{}/{}_angular.png".format(img_id, img_id)).astype(np.float)
    gcgan_img = cv2.imread("../../Downloads/fig3/{}/{}_gcgan.png".format(img_id, img_id)).astype(np.float)
    img = cv2.imread("../../Downloads/fig3/{}/{}_original.png".format(img_id, img_id)).astype(np.float)

    diff_cifr = np.abs(cifr_img - img) / residuality
    diff_ifrnet = np.abs(ifrnet_img - img) / residuality
    diff_angular = np.abs(angular_img - img) / residuality
    diff_gcgan = np.abs(gcgan_img - img) / residuality
    diffs = [diff_cifr, diff_ifrnet, diff_gcgan, diff_angular]
    return diffs


def plot_error_maps(diffs, names):
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for i, ax in enumerate(axes.flat):
        if i < 4:
            ax.set_title(names[i])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(diffs[i], vmin=0, vmax=1)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.05)
    plt.savefig("../../Downloads/residuals.png")
    plt.show()


if __name__ == '__main__':
    diffs_lst = list()
    for img_id in img_ids:
        diffs = extract_error_map(img_id, residuality)
        diffs_lst += diffs
    plot_error_maps(diffs_lst, diff_names)
