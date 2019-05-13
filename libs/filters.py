import cv2
import numpy as np
from matplotlib import pyplot as plt
from libs.utility import *


def mean_filter(image_path, kernel):
    img = cv2.imread(image_path, 0)
    filtered = cv2.blur(img, kernel)
    compare_images(img, "Original", filtered, "Mean Filter")


def median_filter(image_path, kernel):
    img = cv2.imread(image_path, 0)
    filtered = cv2.medianBlur(img, kernel)
    compare_images(img, "Original", filtered, "Mean Filter")


def sobel_filter(image_path, kernel):
    img = cv2.imread(image_path, 0)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    compare_images(img, "Original", sobelx, "Sobel X")
    compare_images(img, "Original", sobely, "Sobel Y")
    compare_images(img, "Original", sobelx + sobely, "Sobel X + Y")


def low_pass_n_times(image_path, kernel, n):
    img = cv2.imread(image_path, 0)

    imgs = [img]

    first = False

    for i in range(0, n):
        last = imgs[i]
        blur = cv2.GaussianBlur(last, kernel, 0)
        #compare_images(last, "Low-Pass Applied " + str(i) + " times", blur, "Low-Pass Applied " + str(i+1) + " times")
        imgs.append(blur)

        difference = cv2.subtract(last, blur)
        result = not np.any(difference)
        if result and (not first):
            print("Identical! - " + str(i + 1))
            first = True
    compare_images(img, "Original", imgs[len(imgs) - 1], "100 Times low-Pass")

