from libs.utility import *
import cv2
from matplotlib import pyplot as plt
from libs.regiongrowing import *

def run():
    ex = input("Input the exercise to execute: ")

    if ex == "2":
        ex2()
    elif ex == "3":
        ex3()
    elif ex == "5":
        ex5()
    elif ex == "6":
        ex6()
    elif ex == "7":
        ex7()
    elif ex == "10":
        ex10()


def ex2():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    shanon_ex(img)


def ex5():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    tsallis_ex(img, 0.8)


def ex6():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    tsallis_ex_2x(img, 2.0)


def ex7():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    kernel = np.ones((7,7),np.uint8)

    plt.imshow(img_blur, cmap="binary")
    plt.show()

    x, thresh = tsallis(img_blur, 0.999)

    img_seg = img_blur.copy()

    for i in range(0, len(img_blur)):
        for j in range(0, len(img_blur[0])):
            img_seg[i][j] = 255 if img_seg[i][j] >= thresh else 0


    plt.imshow(img_seg, cmap="binary")
    plt.show()

    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    plt.imshow(opened, cmap="binary")
    plt.show()


def ex10():
    img = cv2.imread("image/japanese-temple7.png", cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    kernel = np.ones((5, 5), np.uint8)

    plt.imshow(img_blur, cmap="binary")
    plt.show()

    x, thresh = tsallis(img_blur, 0.999)

    img_seg = img_blur.copy()

    for i in range(0, len(img_blur)):
        for j in range(0, len(img_blur[0])):
            img_seg[i][j] = 255 if img_seg[i][j] >= thresh else 0

    plt.imshow(img_seg, cmap="binary")
    plt.show()

    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    plt.imshow(opened, cmap="binary")
    plt.show()

