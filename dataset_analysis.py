from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import skdim
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(dataset_name: str, dataset_percentage=None):
    file_path = f"./data/{dataset_name}.csv"

    if Path(file_path).exists():
        try:
            df = pd.read_csv(file_path, header=0, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, header=0, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, header=0, encoding='cp1252')

        df = pd.read_csv(file_path, header=0)
        labels = df['class']
        data = df.drop('class', axis=1)
        if dataset_percentage:
            _, X, _, y = train_test_split(data, labels, test_size=dataset_percentage, stratify=labels, random_state=42)
            return y, X
        else:
            return labels, data
    else:
        return None


def analyze_dataset(dataset_name: str, k: int, n_jobs: int = 1, pre_loaded_dataset=None, dataset_percentage=None):
    if pre_loaded_dataset:
        loaded_dataset = pre_loaded_dataset
    else:
        loaded_dataset = load_dataset(dataset_name, dataset_percentage=dataset_percentage)
    if loaded_dataset:
        labels, data = loaded_dataset
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        knn.fit(data, labels)

        knn_graph = knn.kneighbors_graph()
        graph = nx.from_scipy_sparse_array(knn_graph, create_using=nx.DiGraph)
        in_degrees = dict(graph.in_degree())

        _, indices = knn.kneighbors(return_distance=True)

        lids = skdim.id.lPCA().fit_transform_pw(data, n_neighbors=k, precomputed_knn=indices, n_jobs=n_jobs)

        in_degrees_list = []
        local_intrinsic_dimensionality_list = []

        node_data = {}
        for node, in_degree in in_degrees.items():
            in_degrees_list.append(in_degree)
            local_intrinsic_dimensionality_list.append(lids[node])
            node_data.update({node: (in_degree, lids[node])})

        correlation_coefficient = np.corrcoef(in_degrees_list, local_intrinsic_dimensionality_list)[0, 1]
        Path("./analysis_data").mkdir(exist_ok=True)
        f = open(f"./analysis_data/{dataset_name}.txt", "w+")
        f.write(f"{correlation_coefficient}\n")
        f.write(f"node;in_degree;local_intrinsic_dimensionality\n")
        for node, (in_degree, lid) in node_data.items():
            f.write(f"{node};{in_degree};{lid}\n")
        f.close()
        return correlation_coefficient
    else:
        return False
