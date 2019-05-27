import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy import ndimage, misc
# Imports lista 2
from skimage.filters import threshold_mean, threshold_otsu, threshold_yen


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
    return colors


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

def histogram(h):
    hist = np.array(256 * [0.0])
    for p in h:
        hist[p] = hist[p] + 1

    return hist

def get_entropy(hist, s_func, ent_func):

    # hist = hist / sum(hist)

    max_ent = np.NINF
    for i in range(2, len(hist) - 1):
        h1 = [hist[p] for p in range(0, i)]
        h1_total = np.sum(h1)
        h1 = [0.0] * len(h1) if h1_total == 0.0 else h1 / h1_total
        h2 = [hist[p] for p in range(i, len(hist))]
        h2_total = np.sum(h2)
        h2 = [0.0] * len(h2) if h2_total == 0.0 else h2 / h2_total

        s1 = s_func(h1)
        s2 = s_func(h2)
        # s1 = calc_ent1(h1)
        # s2 = calc_ent1(h2)

        ent = ent_func(s1, s2)
        if ent > max_ent:
            max_ent = ent
            top = i

    return top

def calcEntropy(img, s_func, ent_func):

    h = img.ravel()
    plt.xlabel("Gray Intensity")
    plt.ylabel("Number of occurrences")
    plt.hist(h, 256, [0,  256], color='g')
    plt.show()
    hist = histogram(h)
    return get_entropy(hist, s_func, ent_func)



def calc_ent1(h):
    ent = 0.0
    for i in h:
        if i != 0.0:
            ent += i * np.log(i)
    return - ent


def shannon(img):
    s_func = lambda h: - np.sum([0.0 if h[i] == 0.0 else h[i] * np.log(h[i]) for i in range(0, len(h))])
    ent_func = lambda sp, sq: sp + sq
    hist = histogram(img.ravel())
    thresh = calcEntropy(hist, s_func, ent_func)
    return img > thresh, thresh


def tsallis(img, q):
    s_func = lambda h: (1 - np.sum([h[i]**q for i in range(0, len(h))])) / (q - 1)
    ent_func = lambda sp, sq: sp + sq + (1 - q)*sp*sq
    thresh = calcEntropy(img, s_func, ent_func)
    #dist = dit.Distribution(img.ravel().tolist())
    #thresh = tsallis_entropy(dist, q)
    return img > thresh, thresh


def shanon_ex(img):
    new_img, thresh = shannon(img)
    plt.imshow(new_img, cmap="plasma")

    print("The image threshold is: " + str(thresh))
    plt.show()


def tsallis_ex(img, q):
    new_img, thresh = tsallis(img, q)
    plt.imshow(new_img, cmap="plasma")

    print("The image threshold for q = " + str(q) + " is: " + str(thresh))
    plt.show()

def tsallis_ex_2x(img, q):
    s_func = lambda h: (1 - np.sum([h[i] ** q for i in range(0, len(h))])) / (q - 1)
    ent_func = lambda sp, sq: sp + sq + (1 - q) * sp * sq

    hist = histogram(img.ravel())
    thresh1 = get_entropy(hist, s_func, ent_func)

    #new_hist = [hist[i] for i in range(thresh1, len(hist))]
    if thresh1 > 127:
        new_hist = [hist[i] for i in range(0, thresh1)]
    else:
        new_hist = [hist[i] for i in range(thresh1, len(hist))]
    thresh2 = get_entropy(new_hist, s_func, ent_func)

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if img[i][j] >= max(thresh2, thresh1):
                img[i][j] = 255
            elif img[i][j] >= min(thresh2, thresh1):
                img[i][j] = 120
            else:
                img[i][j] = 0

    print("Thesh 1 = " + str(thresh1) + " Thresh 2 = " + str(thresh2))

    plt.imshow(img, cmap="plasma")

    plt.show()

