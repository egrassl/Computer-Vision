import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy import ndimage, misc
# Imports lista 2
from skimage.filters import threshold_mean, threshold_otsu, threshold_yen, threshold_adaptive


def compare_images(img1, title1, img2, title2):
    plt.subplot(211), plt.imshow(img1, cmap='gray')
    plt.title(title1), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(img2, cmap='gray')
    plt.title(title2), plt.xticks([]), plt.yticks([])
    plt.show()


def img_histogram(img_gray, title):
    # Plots Initial histogram
    colors = img_gray.flatten()
    plt.title(title)
    plt.xlabel("Gray Intensity")
    plt.ylabel("Number of occurrences")
    plt.hist(colors, 256, [0, 256], color='g')
    plt.show()


def get_gradient(img_gray, ksize, p_sobel):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)

    if p_sobel:
        compare_images(sobelx, "Sobel X", sobely, "Sobel Y")

    sobelx = sobelx.flatten()
    sobely = sobely.flatten()

    n = len(sobelx)

    thetas = []

    for i in range(0, n):
        theta = 0.0
        if sobelx[i] == 0 and sobely[i] != 0:
            if sobely[i] < 0:
                theta = -90.0
            else:
                theta = 90.0
            thetas.append(theta)
        else:
            x = sobelx[i]
            y = sobely[i]
            # k = y / x
            # if y != 0:
            # breakpoint()
            if x != 0:
                theta = np.arctan2(y, x)
                theta = np.degrees(theta)
                if theta < 0:
                    theta = 360 + theta
                thetas.append(theta)

    return np.array(thetas, np.float)


def image_gradient(img_gray, title, ksize = 5, p_sobel = False):
    r = [0, 360]
    values = 360

    thetas = get_gradient(img_gray, ksize, p_sobel)
    plt.title(title)
    plt.xlabel("Direction")
    plt.ylabel("Number of occurrences")
    plt.hist(thetas, values, r, color='g')
    plt.show()

    fd, hog_image = hog(img_gray, orientations=10, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=False)

    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()

# Calculates image gradient only for positive theta values
def absolute_image_gradient(img_gray, title, ksize = 5, p_sobel = False):
    r = [0, 180]

    thetas = get_gradient(img_gray, ksize, p_sobel)

    for i in range(0, len(thetas)):
        theta = thetas[i]
        if theta > 180:
            thetas[i] = theta - 180

    plt.title(title)
    plt.xlabel("Direction")
    plt.ylabel("Number of occurrences")
    plt.hist(thetas, 180, r, color='g')
    plt.show()

    fd, hog_image = hog(img_gray, orientations=10, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=False)

    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()

def laplacian_dist(img_gray, title, ksize = 5):

    laplace = ndimage.filters.laplace(img_gray, mode='wrap')
    laplace = np.array(laplace).ravel()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.hist(laplace, 'auto', facecolor='green', histtype='bar', align='left', rwidth=0.8)
    ax.set_title(title)

    plt.show()


def gray_scale(img):
    # Reads image, resize it, and convert it to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


# ====== LISTA 2 ======

def sliding_window(image, stepSize, windowSize):
    image = np.asarray(image, dtype=np.float32)
    (winW, winH) = windowSize

    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            window = image[y:y + windowSize[1], x:x + windowSize[0]]

            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            hist = np.histogram(window.ravel(), bins=range(256), density=True)

            ent = 0.0
            for p in hist[0]:
                if p != 0.0:
                    ent += p * np.log(p)

            image[y:y + windowSize[1], x:x + windowSize[0]] = -ent

    return image


def shannon(img, length):
    ###################################### Shannon ######################################
    print('Shannon')

    clone = img.copy()
    (winW, winH) = (length, length)

    img_new = sliding_window(clone, stepSize=length, windowSize=(winW, winH))[:-length]

    img_new = img_new[:-length]
    thresh = threshold_mean(img_new)
    img_new = img_new > thresh

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.imshow(img_new, cmap = 'viridis')
    ax.set_title('Shannon')
    ax.axis('off')
    plt.show()

    return img_new
