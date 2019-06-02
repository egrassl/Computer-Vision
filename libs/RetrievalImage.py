import cv2
import numpy as np
from libs.utility import *
from skimage.feature import hog


class RetrievalImage(object):

    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.histogram = self.get_hist(self.img_gray)
        self.gradient_histogram = np.array(self.get_gradient_hist(self.img_gray))

    @staticmethod
    def get_hist(img_gray):
        #hist = ut_histogram(img_gray.ravel())
        hist = ut_histogram(img_gray)
        #sum = np.sum(hist)
        hist /= np.sum(hist)
        #sum = np.sum(hist)
        return hist

    @staticmethod
    def get_gradients(img_gray):
        fd = hog(img_gray, orientations=10, pixels_per_cell=(16, 16), multichannel=False, feature_vector=True)
        a = fd[0]
        return fd

    @staticmethod
    def get_gradient_hist(img):
        angles = []

        img = np.array(img, dtype=np.float32)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        _, ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

        ang = ang.ravel()
        angles += ang.tolist()

        hist = np.histogram(angles, bins=range(10), density=True)

        return hist[0].tolist()
