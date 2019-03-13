import cv2
import numpy as np
from matplotlib import pyplot as plt

plot_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

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

def kmeans_segmentation(image_path, img_size, blur_kernel, k):
    # Reads image, resize it, and convert it to grayscale
    img_gray = resize_and_grayscale(image_path, img_size)
    if blur_kernel != 0:
        img_gray = cv2.medianBlur(img_gray, blur_kernel)

    # Plots Initial histogram
    colors = img_gray.flatten()
    plt.title("Original Color Histogram")
    plt.xlabel("Gray Intensity")
    plt.ylabel("Number of occurrences")
    plt.hist(colors, 256, [0, 256], color='g')
    plt.show()

    # Executes KMeans
    crit = (cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    center_init = cv2.KMEANS_RANDOM_CENTERS

    k_colors = np.array([[c] for c in colors], np.float32)
    compactness, labels, centers = cv2.kmeans(k_colors, k, None, crit, 10, center_init)
    centers = np.array(centers, np.int)

    # Calculates K value text to plot above the bars
    points = [0] * len(centers)
    for i in range(0, len(centers)):
        counter = 0
        for j in range(0, len(colors)):
            if centers[i][0] == colors[j]:
                counter = counter + 1
        points[i] = counter

    plt.title("Color Histogram With Clusters")
    plt.xlabel("Gray Intensity")
    plt.ylabel("Number of occurrences")
    for i in range(0, len(centers)):
        group_index = [i for i in np.where(labels.flatten() == i)]
        group = [colors[index] for index in group_index[0]]
        plt.hist(group, 256, [0, 256], color=plot_colors[i % len(plot_colors)])
        # Plots K value text in the same X as the center and above the bars
        plt.text(centers[i][0], points[i]*2.5, "K = " + str(i + 1) + "; Value = " + str(centers[i][0]),
                 color=plot_colors[i % len(plot_colors)])
    plt.show()

    # Shows Images
    new_img = [[centers[label]] for label in labels]
    new_img = np.reshape(new_img, (img_size[1], img_size[0]))
    new_img = np.uint8(new_img)
    cv2.imshow("KMeans segmented", new_img)
    cv2.waitKey(0)

def color_kmeans_segmentation(image_path, img_size, blur_kernel, k):
    # Reads image, resize it, and convert it to grayscale
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    cv2.imshow("Original Image", img_resized)
    cv2.waitKey(0)

    if blur_kernel != 0:
        img_resized = cv2.medianBlur(img_resized, blur_kernel)

    # Plots Initial histogram
    colors = img_resized.reshape((-1, 3))

    colors = np.float32(colors)

    # Executes KMeans
    crit = (cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    center_init = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(colors, k, None, crit, 10, center_init)

    centers = np.uint8(centers)
    new_img = centers[labels.flatten()]
    new_img = new_img.reshape(img_resized.shape)

    cv2.imshow("Segmented Image", new_img)
    cv2.waitKey(0)
