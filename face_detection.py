import os
import numpy as np
import cv2
from face_recognition import Preprocessor, load_images


class HaarFeature(object):

    """Class representing a Haar-Like feature."""  

    def __init__(self, feature_type, position, size, weight = 1/255):
        """
        Initialize a Haar-Like feature.

        Args:
            feature_type (str): Type of the Haar feature (e.g., 'two_horizontal', 'three_vertical').
            position (tuple): Top-left corner as (x, y).
            size (tuple): Size of the feature.
            weight (float, optional): Weight of the feature. Defaults to 1/255.
        """
        self._feature_type = feature_type  # Type of the Haar feature (e.g., 'two_horizontal', 'three_vertical')
        self._position = position          # top-left corner as (x, y) 
        self._size = size                  
        self._weight = weight 

    @property
    def feature_type(self):
        return self._feature_type

    @feature_type.setter
    def feature_type(self, value):
        self._feature_type = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value
    
    def compute_value(self, integral_image):
        """
        Compute the value of the Haar-like feature using the integral image.
        :param integral_image: IntegralImage object containing the integral representation of the image.
        :return: Value of the Haar-like feature.
        """
        x, y = self.position
        width, height = self.size

        if self.feature_type == 'two_horizontal':
            
            top_left = (x, y)
            bottom_right_top = (x + width // 2 - 1, y + height - 1)
            bottom_left = (x, y + height)
            bottom_right_bottom = (x + width // 2 - 1, y + 2 * height - 1)

            top_sum = integral_image.compute_sum(top_left, bottom_right_top)
            bottom_sum = integral_image.compute_sum(bottom_left, bottom_right_bottom)

            return self.weight * (top_sum - bottom_sum)

        elif self.feature_type == 'three_vertical':
           
            top_left = (x, y)
            middle_left = (x, y + height)
            bottom_left = (x, y + 2 * height)
            top_right = (x + width - 1, y)
            middle_right = (x + width - 1, y + height)
            bottom_right = (x + width - 1, y + 2 * height)

            top_sum = integral_image.compute_sum(top_left, top_right)
            middle_sum = integral_image.compute_sum(middle_left, middle_right)
            bottom_sum = integral_image.compute_sum(bottom_left, bottom_right)

            return self.weight * (top_sum - middle_sum + bottom_sum)

 



class IntegralImage(object):

    """Class for computing integral images."""

    def __init__(self, image):
        """
        Initialize IntegralImage with an input image.

        Args:
            image (ndarray): Input image as a NumPy array.
        """
        self.integral_image = self._compute_integral_image(image)

    def _compute_integral_image(self, image):
        """
        Compute the integral image of the input image.

        Args:
            image (ndarray): Input image as a NumPy array.

        Returns:
            ndarray: Integral image.
        """
        integral_image = np.zeros_like(image)
        integral_image[0][0] = image[0][0]

        for j in range(1,len(image[0])):
            integral_image[0][j] = image[0][j]+ integral_image[0][j-1]
            
        for i in range(1, len(image)):
            integral_image[i][0] = image[i][0] + integral_image[i - 1][0]
    
        for i in range(1, len(image)):
            for j in range(1, len(image[0])):
                integral_image[i][j] = image[i][j] + integral_image[i - 1][j] + \
                                        integral_image[i][j - 1] - integral_image[i - 1][j - 1]

        return integral_image

    def compute_sum(self, top_left, bottom_right):
        """
        Computes the sum of pixel values within the rectangle defined by top_left and bottom_right.
        :param top_left: Tuple (x, y) representing the coordinates of the top-left corner of the rectangle.
        :param bottom_right: Tuple (x, y) representing the coordinates of the bottom-right corner of the rectangle.
        :return: Sum of pixel values within the rectangle.
        """
        x1, y1 = top_left
        x2, y2 = bottom_right

        A = self.integral_image[y1-1][x1-1] if x1 > 0 and y1 > 0 else 0
        B = self.integral_image[y1-1][x2] if y1 > 0 else 0
        C = self.integral_image[y2][x1-1] if x1 > 0 else 0
        D = self.integral_image[y2][x2]

        return D - B - C + A


class AdaBoost:
    def __init__(self, num_classifiers):
        self.num_classifiers = num_classifiers
        self.classifiers = []

    def train(self, images, labels):
        weights = np.ones(len(images)) / len(images)  # Initialize weights

        for _ in range(self.num_classifiers):
            classifier = self.train_weak_classifier(images, labels, weights)
            error = 0.0

            for i in range(len(images)):
                if classifier.classify(images[i]) != labels[i]:
                    error += weights[i]

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update weights
            for i in range(len(images)):
                if classifier.classify(images[i]) != labels[i]:
                    weights[i] *= np.exp(alpha)
                else:
                    weights[i] *= np.exp(-alpha)

            weights /= np.sum(weights)

            self.classifiers.append((classifier, alpha))

    def train_weak_classifier(self, images, labels, weights):
        best_feature = None
        best_error = float('inf')
        image_width, image_height = (images.shape[2], images.shape[1])
        for feature_type in ['two_horizontal', 'three_vertical']:  
            for x in range(image_width):
                for y in range(image_height):
                    for w in range(1, image_width - x + 1):
                        for h in range(1, image_height - y + 1):
                            feature = HaarFeature(feature_type, (x, y), (w, h))
                            error = 0.0
                            for i in range(len(images)):
                                if feature.compute_value(images[i]) != labels[i]:
                                    error += weights[i]

                            if error < best_error:
                                best_feature = feature
                                best_error = error

        return best_feature

    def classify(self, image):
        total = 0.0
        for classifier, alpha in self.classifiers:
            total += alpha * classifier.compute_value(image)
        return 1 if total >= 0.5 * sum(alpha for _, alpha in self.classifiers) else 0
