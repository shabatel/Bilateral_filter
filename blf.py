import sys

import numpy as np
import cv2
import math
from multiprocessing import Pool

FILTER_DIAMETER = 3
SIGMA_R = 30  # photometric
SIGMA_D = 3  # geometric
PROC_NUM = 4
IMG_OUTPUT_NAME = sys.argv[2]


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(x ** 2) / (2 * (sigma ** 2)))


def apply_bilateral_filter(source, filtered_image, row, col, diameter, sigma_r, sigma_d):
    hl = int(diameter / 2)
    i_filtered = 0
    wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_row = int(row - (hl - i))  # neighbours in diameter
        if 0 <= neighbour_row < len(source):  # within the img range
            while j < diameter:
                neighbour_col = int(col - (hl - j))
                if 0 <= neighbour_col < len(source):
                    gauss_r = gaussian(abs(source[row][col] - source[neighbour_row][neighbour_col]), sigma_r)
                    gauss_d = gaussian(distance(row, col, neighbour_row, neighbour_col), sigma_d)
                    w = gauss_r * gauss_d
                    i_filtered += w * source[neighbour_row][neighbour_col]
                    wp += w
                j += 1
        i += 1
    i_filtered = i_filtered / wp
    filtered_image[row][col] = int(i_filtered)


def bilateral_filter_own(args):
    source, filter_diameter, sigma_r, sigma_d = args
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_r, sigma_d)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":
    img = cv2.imread(str(sys.argv[1]))
    if len(img.shape) != 3:  # process grey img
        grey_img = bilateral_filter_own(args=[img, FILTER_DIAMETER, SIGMA_R, SIGMA_D])
        cv2.imwrite(IMG_OUTPUT_NAME, grey_img)
    else:  # process colored img
        blue, green, red = cv2.split(img)
        pool = Pool(processes=PROC_NUM)

        # split the img to 3 colors and run the filter on each one of them by other proc simultaneously
        future_blue, future_green, future_red = pool.map(bilateral_filter_own,
                                                         [(blue, FILTER_DIAMETER, SIGMA_R, SIGMA_D),
                                                          (green, FILTER_DIAMETER, SIGMA_R, SIGMA_D),
                                                          (red, FILTER_DIAMETER, SIGMA_R, SIGMA_D)])
        mered_own = cv2.merge((future_blue, future_green, future_red))  # merge back to one img
        cv2.imwrite(IMG_OUTPUT_NAME, mered_own)
