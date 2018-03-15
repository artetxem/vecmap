"""
Utility methods for alternative matching algorithms.
"""

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure


def otsu(hist, xp):
    """
    skimage.filters.threshold_otsu(camera)
    :param hist:
    :param xp: numpy or cupy
    :return:
    """
    bin_centers = xp.arange(len(hist))
    hist = hist.astype(float)
    # print(hist)

    # class probabilities for all possible thresholds
    weight1 = xp.cumsum(hist)
    weight2 = xp.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = xp.cumsum(hist * bin_centers) / weight1
    mean2 = (xp.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = xp.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


if __name__ == '__main__':

    camera = data.camera()
    val = filters.threshold_otsu(camera)
    print(val)

    hist, bins_center = exposure.histogram(camera)

    plt.figure(figsize=(9, 4))
    plt.subplot(131)
    plt.imshow(camera, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(camera < val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')

    plt.tight_layout()
    plt.show()
