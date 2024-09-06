import networkx as nx
import numpy as np
import skdim
from sklearn.neighbors import KNeighborsClassifier

import dataset_analysis


def calculate_bad_hubness_weights(X, Y, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    if not pre_trained_classifier:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        knn.fit(X, Y)
    else:
        knn = pre_trained_classifier
    knn_graph = knn.kneighbors_graph()
    graph = nx.from_scipy_sparse_array(knn_graph, create_using=nx.DiGraph)
    bad_hubness = [0] * len(Y)
    for (node, nearest_neighbour) in list(graph.edges):
        if Y[node] != Y[nearest_neighbour]:
            bad_hubness[nearest_neighbour] += 1
    bad_hubness = np.array(bad_hubness)
    mean_bad_hubness = np.mean(bad_hubness)
    bad_hubness_standard_deviation = np.std(bad_hubness)
    mean_subtracted = bad_hubness - mean_bad_hubness
    return np.exp((mean_subtracted / bad_hubness_standard_deviation) * -1)

def calculate_hubness_normalized(X, Y = None, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    hubness = calculate_hubness(X, Y, pre_trained_classifier, n_jobs, n_neighbors)
    hubness_min = np.min(hubness)
    hubness_max = np.max(hubness)
    return (hubness - hubness_min) / (hubness_max - hubness_min)

def calculate_hubness(X, Y = None, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    if not pre_trained_classifier:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        knn.fit(X, Y)
    else:
        knn = pre_trained_classifier
    X = np.array(X)
    knn_graph = knn.kneighbors_graph()
    graph = nx.from_scipy_sparse_array(knn_graph, create_using=nx.DiGraph)
    in_degrees = dict(graph.in_degree())
    in_degrees_list = []
    for node, in_degree in in_degrees.items():
        in_degrees_list.append(in_degree)
    return np.array(in_degrees_list)

def calculate_lid_normalized(X, Y = None, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    lid = calculate_lid(X, Y, pre_trained_classifier, n_jobs, n_neighbors)
    lid_min = np.min(lid)
    lid_max = np.max(lid)
    return (lid - lid_min) / (lid_max - lid_min)

def calculate_lid(X, Y, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    if not pre_trained_classifier:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        knn.fit(X, Y)
    else:
        knn = pre_trained_classifier
    precomputed_knn_arrays = knn.kneighbors(return_distance=True)
    return np.array(skdim.id.MLE().fit_transform_pw(X=X, n_neighbors=n_neighbors, precomputed_knn_arrays=precomputed_knn_arrays, n_jobs=n_jobs))

def calculate_lid_weights(X, Y, pre_trained_classifier = None, n_jobs = 1, n_neighbors = 3):
    lid = calculate_lid(X, Y, pre_trained_classifier, n_jobs, n_neighbors)
    mean_lid = np.mean(lid)
    lid_standard_deviation = np.std(lid)
    mean_subtracted = lid - mean_lid
    return np.exp((mean_subtracted / lid_standard_deviation) * 1)

def calculate_lid_and_bad_hubness_weights(X, Y, k, pre_trained_classifier = None, n_jobs = 1):
    if not pre_trained_classifier:
        knn = KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs)
        knn.fit(X, Y)
    else:
        knn = pre_trained_classifier
    return calculate_bad_hubness_weights(X, Y, knn, n_jobs), calculate_lid_weights(X, Y, k, knn, n_jobs)

if __name__ == '__main__':
    Y, X = dataset_analysis.load_dataset("musk")
    weights = calculate_bad_hubness_weights(X, Y, n_jobs=12)
    print(weights)

