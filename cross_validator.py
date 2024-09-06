import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

from datasets import datasets
from utils import calculate_bad_hubness_weights, calculate_lid_weights
import matplotlib.pyplot as plt

import dataset_analysis


class knn_classifier:
    _name = "knn_classifier"
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train, pre_trained_classifier_lid = None, pre_trained_classifier_hubness = None):
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

    def get_name(self):
        "knn_classifier"

class knn_classifier_hubness_weighted:
    _name = "knn_classifier_hubness_weighted"
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train, pre_trained_classifier_lid = None, pre_trained_classifier_hubness = None):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_bad_hubness_weights(X_train, y_train, n_jobs=12, n_neighbors=self.k, pre_trained_classifier=pre_trained_classifier_hubness)

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

    def get_name(self):
        "knn_classifier_hubness_weighted"

class knn_classifier_lid:
    _name = "knn_classifier_lid"
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train, pre_trained_classifier_lid = None, pre_trained_classifier_hubness = None):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_lid_weights(X_train, y_train, n_jobs=12, n_neighbors=100, pre_trained_classifier=pre_trained_classifier_lid)

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
    _name = "knn_classifier_lid_and_hubness_weight"
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train, pre_trained_classifier_lid = None, pre_trained_classifier_hubness = None):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = calculate_lid_weights(X_train, y_train, n_jobs=12, n_neighbors=100, pre_trained_classifier=pre_trained_classifier_lid) + calculate_bad_hubness_weights(X_train, y_train, n_jobs=12, n_neighbors=self.k, pre_trained_classifier=pre_trained_classifier_hubness)

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

    def get_name(self):
        "knn_classifier_lid_and_hubness_weight"


def cross_validate(X, y, classifiers, n_jobs=1, n_neighbours=3):
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    knn_hubness = KNeighborsClassifier(n_neighbors=n_neighbours, n_jobs=n_jobs)
    knn_lid = KNeighborsClassifier(n_neighbors=100, n_jobs=n_jobs)

    accuracies = {}
    for classifier in classifiers:
        accuracies.update({classifier._name: []})
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn_lid.fit(x_train, y_train)
        knn_hubness.fit(x_train, y_train)

        for classifier in classifiers:
            classifier_accuracies = accuracies.get(classifier._name)
            classifier.fit(x_train, y_train, pre_trained_classifier_lid=knn_lid, pre_trained_classifier_hubness=knn_hubness)

            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            classifier_accuracies.append(accuracy)
            accuracies.update({classifier._name: classifier_accuracies})
    for classifier in classifiers:
        average = np.average(accuracies.get(classifier._name))
        mean = np.mean(accuracies.get(classifier._name))
        accuracy = accuracies.get(classifier._name)
        accuracies.update({classifier._name: (average, mean, accuracy)})
    return accuracies

if __name__ == '__main__':
    timestamp = f"{datetime.datetime.now().timestamp()}".split(".")[0]
    datasets_to_check = {"mfeat-factors":"mfeat-factors", "mfeat-fourier":"mfeat-fourier", "optdigits":"optdigits", "segment":"segment", "spectrometer":"spectrometer", "vehicle":"vehicle"}
    # datasets_to_check = datasets
    for dataset_title, dataset_name in datasets_to_check.items():
        Y, X = dataset_analysis.load_dataset(dataset_name)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        k_values = [60, 80, 100]
        knn_scores = []
        knn_hubness_weighted_scores = []
        knn_lid_weighted_scores = []
        knn_lid_and_hubness_weighted_scores = []

        scores = {}
        for k in k_values:
            classifiers = [
                knn_classifier(k=k),
                knn_classifier_hubness_weighted(k=k),
                knn_classifier_lid(k=k),
                knn_classifier_lid_and_hubness_weight(k=k)
            ]
            accuracy_scores = cross_validate(np.array(X), np.array(Y), classifiers, n_jobs=12)
            knn_score = accuracy_scores.get(classifiers[0]._name)[1]
            knn_hubness_weighted_score = accuracy_scores.get(classifiers[1]._name)[1]

            knn_scores.append(knn_score)
            knn_hubness_weighted_scores.append(knn_hubness_weighted_score)

            knn_lid_weighted_score = accuracy_scores.get(classifiers[2]._name)[1]
            knn_lid_and_hubness_weighted_score = accuracy_scores.get(classifiers[3]._name)[1]

            knn_lid_weighted_scores.append(knn_lid_weighted_score)
            knn_lid_and_hubness_weighted_scores.append(knn_lid_and_hubness_weighted_score)
            scores.update({k: accuracy_scores})
        Path(f"./plots/plots_{timestamp}").mkdir(exist_ok=True)
        with open(f'plots/plots_{timestamp}/{dataset_name}.csv', "w+") as f:
            header = "classifiers"
            data = {}
            for k, accuracy_scores in scores.items():
                header += f";{k}"
                for classifier, (avg, mean, scores) in accuracy_scores.items():
                    if classifier in data.keys():
                        means = data.get(classifier)
                        means.append(mean)
                        data.update({classifier: means})
                    else:
                        data.update({classifier: [mean]})

            f.write(header+"\n")
            for classifier, means in data.items():
                row = f"{classifier}"
                for mean in means:
                    row += f";{mean}"
                f.write(row+"\n")

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, knn_scores, label='k-NN Classifier', marker='o')
        plt.plot(k_values, knn_hubness_weighted_scores, label='k-NN Hubness Weighted Classifier', marker='o')
        plt.plot(k_values, knn_lid_weighted_scores, label='k-NN LID Weighted Classifier', marker='o')
        plt.plot(k_values, knn_lid_and_hubness_weighted_scores, label='k-NN LID and Hubness Weighted Classifier', marker='o')

        plt.title('Classifier Performance vs. k Value')
        plt.xlabel('k Value')
        plt.ylabel('Score')
        plt.legend()

        plt.savefig(f'plots/plots_{timestamp}/{dataset_name}.png')
        # plt.show()
