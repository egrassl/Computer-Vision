import cv2
import numpy as np

def resize_and_grayscale(image_path, img_size):
    # Reads image, resize it, and convert it to grayscale
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_gray

def limiar_filter(image_path, img_size, blur_kernel, threshold):
    # Reads image, resize it, and convert it to grayscale
    img_gray = resize_and_grayscale(image_path, img_size)
    if blur_kernel != 0:
        img_gray = cv2.medianBlur(img_gray, blur_kernel)

    # Applies thresholding
    r, thres = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    mean = int(np.mean(img_gray))
    r, thres2 = cv2.threshold(img_gray, mean, 255, cv2.THRESH_BINARY)

    # plots Images
    cv2.imshow("Grayscale Image", img_gray)
    cv2.waitKey(0)
    cv2.imshow("Threshold Image (t = " + str(threshold) + ")", thres)
    cv2.waitKey(0)
    cv2.imshow("Gaussian Adaptative Threshold Image (t = " + str(mean) + ")", thres2)
    cv2.waitKey(0)


def adaptative_filter(image_path, img_size, blur_kernel, kernel_size, c):

    # Reads image, resize it, and convert it to grayscale
    img_gray = resize_and_grayscale(image_path, img_size)
    if blur_kernel != 0:
        img_gray = cv2.medianBlur(img_gray, blur_kernel)

    # Applies mean and gaussian adaptative thresholding
    mean_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, c)
    gauss_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            kernel_size, c)

    # plots Images
    cv2.imshow("Grayscale Image", img_gray)
    cv2.waitKey(0)
    cv2.imshow("Mean Adaptative Threshold Image", mean_threshold)
    cv2.waitKey(0)
    cv2.imshow("Gaussian Adaptative Threshold Image", gauss_threshold)
    cv2.waitKey(0)

