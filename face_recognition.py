import numpy as np
import cv2
import os


PATH = r"G:\my-projects\computer vision\Humans"


class Preprocessor(object):
    def __init__(self,images):
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
        resized_images = []
        for image in self.images:
            resized_image = cv2.resize(image,(self.width,self.height))
            resized_images.append(resized_image)
        self.images = np.array(resized_images)
        return self.images

    def grayscale(self):
        grayscale_images = []
        for image in self.images:
            grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            grayscale_images.append(grayscale_image)
        self.images = np.array(grayscale_images)
        return self.images
    
    def normalize(self):
        self._normalized_images =  self.images/255
        return self.normalized_images
    
    def preprocess(self):
        resized_images = self.resize()
        grayscale_images = self.grayscale()
        normalized_images = self.normalize()
        return self.normalized_images

def load_images(path, n_images):
    images_files = os.listdir(path)
    images = []
    for image in images_files[:n_images]:
        image_path = os.path.join(PATH, image)
        image_data = cv2.imread(image_path)
        images.append(image_data)
    return images, images_files[:n_images]


class StandardScaler(object):

    def __init__(self):
        self._std = None
        self._mean = None

    @property
    def std(self):
        return self._std

    @property
    def mean(self):
        return self._mean
    
    def _fit(self, data):
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("standard scaler has not been fitted yet")
        return (data-self.mean) / self.std

    def fit_transform(self, data):
        self._fit(data)
        return self.transform(data)



class EigenFaces(object):
    
    def __init__(self, n_components):
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
        data_scaled = self.scaler.fit_transform(data)
        cov_matrix = np.cov(data_scaled,rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:,sorted_indices]
        self._components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, data):
        data_scaled = self.scaler.transform(data)
        return np.dot(data_scaled, self.components)

    def fit_transform(self, data):
        self._fit(data)
        return self.transform(data)



class KNeighborsClassifier(object):
    def __init__(self, n_neighbors):
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
        self._x_train = x_train
        self._y_train = y_train
    
    def predict(self,x_test):
        predictions = []
        for x in x_test:
            distances = np.sqrt(np.sum((self.x_train-x)**2,axis=1))
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels =self.y_train[nearest_indices]
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)
    
    def calculate_accuracy(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_true = y_test
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total




################################################################ test
# # images = np.array(images)
# images,labels = load_images(PATH, 20)
# pre_data = Preprocessor(images)
# images = pre_data.preprocess()
# data = np.array([image.flatten() for image in images])
# pca = EigenFaces(n_components=100)
# eigen_faces = pca.fit_transform(data)
# knn = KNeighborsClassifier(n_neighbors=1)
# labels = np.array([i for i in range(20)])
# knn.fit(eigen_faces, labels)

# new_face = images[19]
# new_face_eigen = pca.transform(new_face.flatten())
# predicted_label = knn.predict([new_face_eigen])
# print(predicted_label)

# # x = np.abs(eigen_faces[0].reshape(80,80))
# # cv2.imshow("image", x)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()