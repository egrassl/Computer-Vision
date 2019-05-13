import cv2
import numpy as np
import scipy.misc
from libs.utility import *


def run():
    ex = input("Input the exercise to execute: ")

    if ex == "6":
        ex6()
    elif ex == "7":
        ex7()
    elif ex == "10.7":
        ex10_7()


def ex6():
    img = cv2.imread("image/kindle-page.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = (3, 3)
    imgs = [img]

    img_histogram(img, "Original color histogram")

    # Executes blur filter 100 times
    first = False
    for i in range(0, 100):
        last = imgs[i]
        blur = cv2.GaussianBlur(last, kernel, 0)
        imgs.append(blur)

        difference = cv2.subtract(last, blur)
        result = not np.any(difference)
        if result and (not first):
            print("Identical! - " + str(i + 1))
            first = True
    compare_images(img, "Original", imgs[len(imgs) - 1], "100 Times low-Pass")
    img_histogram(imgs[len(imgs) - 1], "Color histogram after 100 times blur")


def ex7():
    img = cv2.imread("image/xadrez.png", cv2.IMREAD_GRAYSCALE)
    laplacian_dist(img, "Laplace Histogram")
    cv2.imshow("Ex7", img)
    cv2.waitKey()
    image_gradient(img, "Image Gradient", 3, False)
    absolute_image_gradient(img, "Absolute Image Gradient", 3, False)


def ex10_7():
    img = cv2.imread("image/kindle-page.jpg", cv2.IMREAD_GRAYSCALE)
    image_gradient(img, "Image Gradient")
    absolute_image_gradient(img, "Absolute Image Gradient")





