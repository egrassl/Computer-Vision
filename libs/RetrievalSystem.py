import os
import cv2
import numpy as np
import re
from libs.RetrievalImage import *
from natsort import natsorted


class RetrievalSystem(object):

    def __init__(self, dataset_dir, labels, comp_func):
        self.images = self.get_images_from_dataset(dataset_dir, labels)
        self.comp_func = comp_func
        self.labels_quantities = self.get_labels_quantites(self.images)

    @staticmethod
    def get_labels_quantites(images):
        labels = {}

        for image in images:
            if not image.label in labels.keys():
                labels[image.label] = 0

        for image in images:
            labels[image.label] += 1

        return labels


    @staticmethod
    def get_images_from_dataset(dataset_dir, labels):

        image_names = os.listdir(dataset_dir)
        image_names = natsorted(image_names)
        images_path = [dataset_dir + "/" + image_name for image_name in image_names]

        if len(images_path) != len(labels):
            raise Exception("Wrong dataset and labels dimensions")

        images = np.array([RetrievalImage(images_path[i], labels[i]) for i in range(0, len(images_path))])

        return images

    def get_image(self, image_path):
        for i in range(0, len(self.images)):
            if self.images[i].path == image_path:
                return self.images[i]
        raise Exception("Image not found in dataset")

    def retrieve_images(self, image_path):
        image = self.get_image(image_path)
        comparison_results = [self.comp_func(image, r_image) for r_image in self.images]
        order = np.argsort(comparison_results)

        sorted_images = self.get_new_array_order(self.images, order)
        return self.calc_pr_curve(sorted_images, image.label)

    @staticmethod
    def get_new_array_order(array, order):
        return [array[index] for index in order]


    def calc_pr_curve(self, retrieve_images, desired_label):
        n_labels = self.labels_quantities[desired_label]

        recall = np.empty(n_labels + 1)
        precision = np.empty(n_labels + 1)

        recall[0] = 0.0
        precision[0] = 1.0

        i = 1
        j = 1
        for image in retrieve_images:
            if image.label == desired_label:
                recall[j] = j/n_labels
                precision[j] = j / i
                j += 1
            i += 1

        return precision, recall

    def retrieve_for_iamges(self, images):
        precisions = []
        recalls = []
        for image in images:
            precision, recall = self.retrieve_images(image)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls
