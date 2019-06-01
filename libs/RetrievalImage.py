import cv2
import numpy as np
from libs.utility import *

class RetrievalImage(object):

    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.histogram = self.get_hist(self.img_gray)

    @staticmethod
    def get_hist(img_gray):
        #hist = ut_histogram(img_gray.ravel())
        hist = ut_histogram(img_gray)
        #sum = np.sum(hist)
        hist /= np.sum(hist)
        #sum = np.sum(hist)
        return hist
