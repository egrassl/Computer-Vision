from libs.utility import *
import cv2
from matplotlib import pyplot as plt
from libs.regiongrowing import *
from scipy.spatial.distance import directed_hausdorff
from sklearn import metrics

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
    elif ex == "12":
        ex12()


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

    print("K = " + str(thresh))

    plt.imshow(img_seg, cmap="binary")
    plt.show()

    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    plt.imshow(opened, cmap="binary")
    plt.show()


def ex12():
    img1 = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("image/japanese-temple7.png", cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (180, 142), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (180, 142), interpolation=cv2.INTER_AREA)

    plt.imshow(img1, cmap="gray")
    plt.show()

    dist = d_kl(img1, img2)

    print("A distância KL das duas imagens é: " + str(dist))

    dist = directed_hausdorff(img1, img2)[0]

    print("A distância de Hausdorf é: " + str(dist))

    l = max([len(img1.ravel()), len(img2.ravel())])
    a1 = np.pad(img1.ravel(), (0, l - len(img1.ravel())), 'constant')
    a2 = np.pad(img2.ravel(), (0, l - len(img2.ravel())), 'constant')

    dist = euclidean(a1.ravel(), a2.ravel())
    print("A distância Euclidiana é: " + str(dist))

    dist = pdm_dist(img1, img2)
    print("A distância PDM é: " + str(dist))


def ex13():
    print("Hi")
