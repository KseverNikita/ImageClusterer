import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.decomposition import PCA
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image

class ImageClusterer:
    def __init__(self, 
                 image: np.ndarray, # Input image 
                 algorithm: str = "Kmeans", # Algorithm with will be used to make clustering
                 number_of_classes: int = 6, # Desired number of clusters
                 color_mode : str = "rgb", # Color format of image ("hsv" or "rgb")
                 number_of_components : int  = 3, # Number of components in PCA algorithm 
                 use_only_hs: bool = False, # Flag to use only Hue and Saturation components in HSV format of image
                 ):
        super().__init__()
        
        assert len(image.shape) == 3 and image.shape[2] == 3, "Wrong shape of the input image, must be (M, N, 3)"
        
        self.mode = color_mode
        self.number_of_classes = number_of_classes
        self.algorithm = algorithm
        
        if (image.max() > 1):
            self.image = (image / 255.0).astype(np.float32)
        
        if (color_mode == "hsv" and use_only_hs):
            self.image = rgb_to_hsv(self.image)[:, :, [0, 1]] # Use only h and s values
        elif (color_mode == "rgb"):
            self.image = self.image
        else:
            self.image = rgb_to_hsv(self.image)
            
        self.pixels = self.image.reshape((self.image.shape[0] * self.image.shape[1], self.image.shape[2])) # Squeeze image to shape = (N * M,  C) 

        if (number_of_components  != 3):
            self.pixels = PCA(number_of_components).fit_transform(self.pixels)
            
        self.init_algorithm(number_of_classes)
        
    def plot_image_rgb(self, figsize : tuple =(10, 10)):
        figure, ax = plt.subplots(figsize=figsize)
        if (self.mode == 'hsv'):
            ax.imshow(hsv_to_rgb(self.image))
        else:
            ax.imshow(self.image)
        plt.axis('off')
        plt.show();
         
    def init_algorithm(self, n_clusters : int):
        if (self.algorithm == "Kmeans"):
            self.algo = cluster.KMeans(n_clusters=n_clusters)
            
    def predict(self, n_clusters : int = None):
        if (n_clusters):
            self.init_algorithm(n_clusters)
            
        self.algo.fit(self.pixels)
        mask = self.algo.labels_.reshape((self.image.shape[0], self.image.shape[1]))
        assert mask.shape == self.image[:, :, 0].shape, "Mask shape and image shape must be equal"
        if (n_clusters is None):
            assert np.unique(mask).shape[0] == self.number_of_classes, "Number of clusters must be equal to number_of_classes parameter"
        return mask