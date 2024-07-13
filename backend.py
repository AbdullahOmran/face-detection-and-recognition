import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

PATH = r"G:\my-projects\computer vision\task5\faces1"


class Preprocessor(object):

    """Class for preprocessing images."""

    def __init__(self,images):
        """
        Initialize Preprocessor with a list of images.

        Args:
            images (list): List of input images.
        """
        self._images = images
        self._width = 60
        self._height = 60
        self._normalized_images = None
    
    @property
    def images(self):
        return self._images
    
    @images.setter
    def images(self, value):
        self._images = value
        
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value

    @property
    def normalized_images(self):
        return self._normalized_images

    def resize(self):
        """
        Resize the images to the specified width and height.

        Returns:
            ndarray: Resized images.
        """
        resized_images = []
        for image in self.images:
            resized_image = cv2.resize(image,(self.width,self.height))
            resized_images.append(resized_image)
        self.images = np.array(resized_images)
        return self.images

    def grayscale(self):
        """
        Convert the images to grayscale.

        Returns:
            ndarray: Grayscale images.
        """
        grayscale_images = []
        for image in self.images:
            grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            grayscale_images.append(grayscale_image)
        self.images = np.array(grayscale_images)
        return self.images
    
    def normalize(self):
        """
        Normalize the pixel values of the images to the range [0, 1].

        Returns:
            ndarray: Normalized images.
        """
        self._normalized_images =  self.images/255
        return self.normalized_images
    
    def preprocess(self):
        """
        Preprocess the images by resizing, converting to grayscale, and normalizing.

        Returns:
            ndarray: Preprocessed images.
        """
        resized_images = self.resize()
        grayscale_images = self.grayscale()
        normalized_images = self.normalize()
        return self.normalized_images

def load_images(path, n_persons, n_images, train, test):
    """
    Load images from the specified directory.

    Args:
        path (str): Path to the directory containing the images.
        n_images (int): Number of images to load.

    Returns:
        tuple: A tuple containing a list of loaded images and a list of corresponding file names.
    """
    train = int(train * n_images)
    test = int(test * n_images)
    images_folders = os.listdir(path)
    image_folder = []
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []
    for image_folder in images_folders[:n_persons]:
        images_path = os.listdir(os.path.join(path, image_folder))
        for image in images_path[:train]:
            image_path = os.path.join(os.path.join(path, image_folder), image)
            image_data = cv2.imread(image_path)
            train_set.append(image_data)
            train_labels.append(int(image_folder))
        for image in images_path[train:train+test]:
            image_path = os.path.join(os.path.join(path, image_folder), image)
            image_data = cv2.imread(image_path)
            test_set.append(image_data)
            test_labels.append(int(image_folder))
    return train_set, train_labels, test_set, test_labels

def get_ROC_curve(y_true, y_scores, class1):
        y_true_binary = np.zeros_like(y_true)
        y_scores_binary = np.zeros_like(y_scores)
        for i in range(len(y_true_binary)):
            if y_true[i] == class1:
                y_true_binary[i] = 1
        for i in range(len(y_scores_binary)):
            if y_scores[i] == class1:
                y_scores_binary[i] = 1
        y_true = y_true_binary
        y_scores = y_scores_binary
        sorted_indices = np.argsort(y_scores)[::-1]
        y_scores_sorted = y_scores[sorted_indices]
        y_true_sorted = y_true[sorted_indices]
        num_positive = np.sum(y_true_sorted==1)
 
        num_negative = len(y_true) - num_positive 
        tp_count = 0
        fp_count = 0
        for score, true_label in zip(y_scores_sorted, y_true_sorted):
            if true_label == 1 and score == 1:
                tp_count += 1
            if score ==1 and true_label == 0:
                fp_count += 1
        
        tpr = tp_count / num_positive 
        fpr = fp_count / num_negative
        
        
        return fpr, tpr
        # y_true_binary = (y_true == class1)
        # y_scores_binary = (y_scores == class1)

        # sorted_indices = np.argsort(y_scores)[::-1]
        # y_scores_sorted = y_scores_binary[sorted_indices]
        # y_true_sorted = y_true_binary[sorted_indices]

        # num_positive = np.sum(y_true_binary)
        # num_negative = len(y_true) - num_positive 

        # tpr_list = []
        # fpr_list = []
        # tp_count = 0
        # fp_count = 0

        # for true_label in y_true_sorted:
        #     if true_label:
        #         tp_count += 1
        #     else:
        #         fp_count += 1

        #     tpr_list.append(tp_count / num_positive)
        #     fpr_list.append(fp_count / num_negative)

        # return np.array(fpr_list), np.array(tpr_list)



class StandardScaler(object):

    """Class for standardizing features by removing the mean and scaling to unit variance."""

    def __init__(self):
        """
        Initialize StandardScaler.
        """
        self._std = None
        self._mean = None

    @property
    def std(self):
        """ndarray: Standard deviation of the features."""
        return self._std

    @property
    def mean(self):
        """ndarray: Mean of the features."""
        return self._mean
    
    def _fit(self, data):
        """
        Compute the mean and standard deviation of the features.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
        """
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

    def transform(self, data):
        """
        Standardize the input data.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("standard scaler has not been fitted yet")
        return (data-self.mean) /self.std

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Standardized data.
        """
        self._fit(data)
        return self.transform(data)



class EigenFaces(object):
    """Class for performing eigenface analysis."""
    
    def __init__(self, n_components):
        """
        Initialize EigenFaces.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self._n_components = n_components
        self._components = None
        self._mean = None
        self._scaler = StandardScaler()

    @property
    def n_components(self):
        return self._n_components
   
    @property
    def components(self):
        return self._components
   
    @property
    def mean(self):
        return self._mean

    @property
    def scaler(self):
        return self._scaler
   
    def _fit(self,data):
        """
        Fit the eigenfaces model to the data.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
        """
        data_scaled = self.scaler.fit_transform(data)
        cov_matrix = data_scaled @ data_scaled.T
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:,sorted_indices]
        transformed_eigenvectors = data_scaled.T @ sorted_eigenvectors
        eigenfaces = transformed_eigenvectors / np.linalg.norm(transformed_eigenvectors, axis=0)
        self._components = eigenfaces[:, :self.n_components]
        # fig, axs = plt.subplots(1, 5, figsize=(16, 10))
        # for i in range(5):
        #     x=np.reshape(self._components[:, i], (60, 60))
        #     axs[i].imshow(x, cmap='gray')
        # plt.show()
      

    def transform(self, data):
        """
        Transform data into eigenface space.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Transformed data.
        """
        data_scaled = self.scaler.transform(data)
        return data_scaled @ self.components
        

    def fit_transform(self, data):
        """
        Fit the model to the data and transform it into eigenface space.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Transformed data.
        """
        self._fit(data)
        return self.transform(data)



class KNeighborsClassifier(object):
    """Class for K-Nearest Neighbors classification."""

    def __init__(self, n_neighbors):
        """
        Initialize KNeighborsClassifier.

        Args:
            n_neighbors (int): Number of neighbors to consider.
        """
        self._n_neighbors = n_neighbors
        self._x_train = None
        self._y_train = None
    
    @property
    def n_neighbors(self):
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self,value):
        self._n_neighbors = value

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    def fit(self,x_train,y_train):
        """
        Fit the classifier to the training data.

        Args:
            x_train (ndarray): Input training data with shape (n_samples, n_features).
            y_train (ndarray): Target training labels with shape (n_samples,).
        """
        self._x_train = x_train
        self._y_train = y_train
    
    def predict(self,x_test, thre=float('inf')):
        """
        Predict labels for test data.

        Args:
            x_test (ndarray): Input test data with shape (n_samples, n_features).

        Returns:
            ndarray: Predicted labels for the input test data.
        """
        predictions = []
        for x in x_test:
            distances = np.sqrt(np.sum((self.x_train-x)**2,axis=1))
            distances = distances[distances< thre]
            if len(distances) == 0:
                predictions.append(-1)
                continue
            nearest_indices = np.argsort(distances)[:self.n_neighbors]     
            nearest_labels =self.y_train[nearest_indices]
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)
    
    def calculate_accuracy(self, x_test, y_test):
        """
        Calculate the accuracy of the classifier on test data.

        Args:
            x_test (ndarray): Input test data with shape (n_samples, n_features).
            y_test (ndarray): Target test labels with shape (n_samples,).

        Returns:
            float: Accuracy of the classifier on the test data.
        """
        y_pred = self.predict(x_test)
        y_true = y_test
        sorted_indices = np.argsort(y_pred)[::-1]
        y_pred = y_pred[sorted_indices]
        y_true = y_true[sorted_indices]
        correct = np.sum(y_true == y_pred)
        
        total = len(y_true)
        return correct / total

   

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




################################################################ test
# images = np.array(images)
fpr = []
tpr = []
for i in range(80,100):
    x_train,y_train, x_test, y_test  = load_images(PATH, 18, 9, 0.7, 0.3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    pre_data = Preprocessor(x_train)
    x_train = pre_data.preprocess()
    pre_data = Preprocessor(x_test)
    x_test = pre_data.preprocess()
    data = np.array([image.flatten() for image in x_train])
    pca = EigenFaces(n_components=100)
    eigen_faces = pca.fit_transform(data)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(eigen_faces, y_train)
    x_test = pca.transform([image.flatten() for image in x_test])
    predicted_labels = knn.predict(x_test,i)
    fp,tp = get_ROC_curve(y_test, predicted_labels,2)
    fpr.append(fp)
    tpr.append(tp)
    # print(f"Accuracy: {knn.calculate_accuracy(x_test, y_test)}")

plt.scatter(fpr,tpr)
plt.show()
# new_face = x_test[0]
# new_face_eigen = pca.transform(new_face.flatten())
# predicted_label = knn.predict([new_face_eigen])
# print(predicted_label, f"y_hat: {y_test[0]}")

# x = np.abs(eigen_faces[0].reshape(60,60))
# x=np.reshape(eigen_faces[:, 0], (60, 60))
# cv2.imshow("image", x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()