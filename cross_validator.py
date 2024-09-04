import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from utils import calculate_bad_hubness_weights, calculate_lid
import matplotlib.pyplot as plt

import dataset_analysis


class knn_classifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class knn_classifier_hubness_weighted:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_bad_hubness_weights(X_train, y_train, n_jobs=12, n_neighbors=self.k)

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_weights = [self.weights[i] for i in k_indices]

        weighted_votes = Counter()
        for label, weight in zip(k_nearest_labels, k_nearest_weights):
            weighted_votes[label] += weight

        most_common = weighted_votes.most_common(1)
        return most_common[0][0]

class knn_classifier_lid:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_lid(X_train, y_train, n_jobs=12, n_neighbors=self.k)

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_weights = [self.weights[i] for i in k_indices]

        weighted_votes = Counter()
        for label, weight in zip(k_nearest_labels, k_nearest_weights):
            weighted_votes[label] += weight

        most_common = weighted_votes.most_common(1)
        return most_common[0][0]

class knn_classifier_lid_and_hubness_weight:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_lid(X_train, y_train, n_jobs=12, n_neighbors=self.k) + calculate_bad_hubness_weights(X_train, y_train, n_jobs=12, n_neighbors=self.k)

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_weights = [self.weights[i] for i in k_indices]

        weighted_votes = Counter()
        for label, weight in zip(k_nearest_labels, k_nearest_weights):
            weighted_votes[label] += weight

        most_common = weighted_votes.most_common(1)
        return most_common[0][0]


def cross_validate(X, y, classifier):
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []

    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

    return np.average(accuracies), np.mean(accuracies), accuracies

if __name__ == '__main__':
    for dataset in ["vehicle"]:
        Y, X = dataset_analysis.load_dataset(dataset)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        k_values = range(1, 11)
        knn_scores = []
        knn_hubness_weighted_scores = []
        knn_lid_weighted_scores = []
        knn_lid_and_hubness_weighted_scores = []

        for k in k_values:
            knn_score = cross_validate(np.array(X), np.array(Y), knn_classifier(k=k))[1]
            knn_hubness_weighted_score = cross_validate(np.array(X), np.array(Y), knn_classifier_hubness_weighted(k=k))[1]

            knn_scores.append(knn_score)
            knn_hubness_weighted_scores.append(knn_hubness_weighted_score)

            # knn_lid_weighted_score = cross_validate(np.array(X), np.array(Y), knn_classifier_lid(k=k))[1]
            # knn_lid_and_hubness_weighted_score = cross_validate(np.array(X), np.array(Y), knn_classifier_lid_and_hubness_weight(k=k))[1]
            #
            # knn_lid_weighted_scores.append(knn_lid_weighted_score)
            # knn_lid_and_hubness_weighted_scores.append(knn_lid_and_hubness_weighted_score)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, knn_scores, label='k-NN Classifier', marker='o')
        plt.plot(k_values, knn_hubness_weighted_scores, label='k-NN Hubness Weighted Classifier', marker='o')
        # plt.plot(k_values, knn_lid_weighted_scores, label='k-NN LID Weighted Classifier', marker='o')
        # plt.plot(k_values, knn_lid_and_hubness_weighted_scores, label='k-NN LID and Hubness Weighted Classifier', marker='o')

        plt.title('Classifier Performance vs. k Value')
        plt.xlabel('k Value')
        plt.ylabel('Score')
        plt.legend()

        # plt.savefig(f'plots/{dataset}.png')
        plt.show()
