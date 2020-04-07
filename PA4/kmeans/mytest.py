import numpy as np
from data_loader import toy_dataset, load_digits
from kmeans import KMeans, KMeansClassifier, get_k_means_plus_plus_center_indices
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import Figure
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.array([[1,2],[3,4],[5,4],[6,3],[7,7]])
n = 5
c = 3
centers = get_k_means_plus_plus_center_indices(n, c, x)
print(centers)
