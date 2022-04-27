# ImageClusterer
Clustering of the rainbow image.

## Repository structure
In this repository you can find class implemetantion of [Image Cluster](https://github.com/KseverNikita/ImageClusterer/blob/main/ImageCluster.py), notebook with examples of using it, with [Experiments](https://github.com/KseverNikita/ImageClusterer/blob/main/ImageClusterer.ipynb) and [rainbow image](https://github.com/KseverNikita/ImageClusterer/blob/main/rainbow.png).


## Class description 

### Input data
Input data must be an image with 3 channels in RGB format. 

### Output data
The 1-channel mask with the same shape will be returned using method predict.

### Initialization parameters
```python
def __init__(self, 
                 image: np.ndarray, # Input image 
                 algorithm: str = "Kmeans", # Algorithm with will be used to make clustering (supported -  ["Kmeans"])
                 number_of_classes: int = 6, # Desired number of clusters
                 color_mode : str = "rgb", # Color format of image (supported - ["hsv", "rgb"])
                 number_of_components : int  = 3, # Number of components in PCA algorithm (supported - [1, 2, 3])
                 use_only_hs: bool = False, # Flag to use only Hue and Saturation components in HSV format of image
                 ):
        super().__init__()
```
