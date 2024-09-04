import numpy as np
import networkx as nx
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import calculate_bad_hubness_weights
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from cross_validator import cross_validate

import dataset_analysis

Y, X = dataset_analysis.load_dataset("musk")
labels = np.array(Y)
data = np.array(X)
class CustomKNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=3, n_jobs=12):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed')

    def get_weight(self, node):
        print(node)
        return self.weights_array[node]

    def fit(self, X, y):
        self.knn.fit(self._precomputed_distance_matrix(X, y), y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = self.knn.predict(X)
        return prediction

    def _precomputed_distance_matrix(self, X, Y):
        weights = calculate_bad_hubness_weights(X, Y, n_jobs=self.n_jobs)
        distance_matrix = pairwise_distances(X, metric='euclidean')
        adjusted_distance_matrix = distance_matrix * weights[:, np.newaxis]
        if np.any(adjusted_distance_matrix < 0) or np.sign(np.min(adjusted_distance_matrix)) == -1.0:
            raise ValueError("Negative values in adjusted distance matrix.")
        return distance_matrix


knn_classifier = CustomKNNClassifier(n_neighbors=60, n_jobs=12)

print(cross_validate(X,Y,knn_classifier))

