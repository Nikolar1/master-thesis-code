from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew
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

        precomputed_knn_arrays = knn.kneighbors(return_distance=True)

        lidsLPCA = skdim.id.lPCA().fit_transform_pw(data, n_neighbors=k, precomputed_knn=precomputed_knn_arrays[1], n_jobs=n_jobs)
        lidsMLENB = skdim.id.MLE(neighborhood_based=True).fit_transform_pw(X=data, n_neighbors=k, precomputed_knn_arrays=precomputed_knn_arrays, n_jobs=n_jobs)
        lidsMLE = skdim.id.MLE().fit_transform_pw(X=data, n_neighbors=k, precomputed_knn_arrays=precomputed_knn_arrays, n_jobs=n_jobs)
        try:
            lidsTLE = skdim.id.TLE().fit_transform_pw(X=data, n_neighbors=k, precomputed_knn_arrays=precomputed_knn_arrays, n_jobs=n_jobs)
        except Exception:
            lidsTLE = [np.nan] * len(in_degrees)

        in_degrees_list = []
        local_intrinsic_dimensionality_list_LPCA= []
        local_intrinsic_dimensionality_list_MLE = []
        local_intrinsic_dimensionality_list_MLENB = []
        local_intrinsic_dimensionality_list_TLE = []
        average_lids = (0,0,0,0)
        i = 0

        node_data = {}
        for node, in_degree in in_degrees.items():
            in_degrees_list.append(in_degree)
            local_intrinsic_dimensionality_list_LPCA.append(lidsLPCA[node])
            local_intrinsic_dimensionality_list_MLE.append(lidsMLE[node])
            local_intrinsic_dimensionality_list_MLENB.append(lidsMLENB[node])
            local_intrinsic_dimensionality_list_TLE.append(lidsTLE[node])
            node_data.update({node: (in_degree, lidsLPCA[node], lidsMLE[node], lidsMLENB[node], lidsTLE[node])})
            average_lids = (
                average_lids[0] + lidsLPCA[node],
                average_lids[1] + lidsMLE[node],
                average_lids[2] + lidsMLENB[node],
                average_lids[3] + lidsTLE[node],
            )
            i += 1

        average_lids = (
            average_lids[0] / i,
            average_lids[1] / i,
            average_lids[2] / i,
            average_lids[3] / i,
        )

        # mean_Nk = np.mean(in_degrees_list)
        # std_Nk = np.std(in_degrees_list, ddof=0)
        # third_moment = np.mean((in_degrees_list - mean_Nk) ** 3)
        # skewness = third_moment / (std_Nk ** 3)
        skewness = skew(np.array(in_degrees_list))
        correlation_coefficient_LPCA = np.corrcoef(in_degrees_list, local_intrinsic_dimensionality_list_LPCA)[0, 1]
        correlation_coefficient_MLE = np.corrcoef(in_degrees_list, local_intrinsic_dimensionality_list_MLE)[0, 1]
        correlation_coefficient_MLENB = np.corrcoef(in_degrees_list, local_intrinsic_dimensionality_list_MLENB)[0, 1]
        correlation_coefficient_TLE = np.corrcoef(in_degrees_list, local_intrinsic_dimensionality_list_TLE)[0, 1]
        Path("./analysis_data").mkdir(exist_ok=True)
        Path(f"./analysis_data/{dataset_name}").mkdir(exist_ok=True)
        f = open(f"./analysis_data/{dataset_name}/{dataset_name}_k{k}.txt", "w+")
        f.write(f"Skewness: {skewness}, Average LID: LPCA: {average_lids[0]}  |||  MLE: {average_lids[1]}  |||  MLENB: {average_lids[2]}  |||  TLE: {average_lids[3]}\n")
        f.write(f"LPCA: {correlation_coefficient_LPCA}  |||  MLE: {correlation_coefficient_MLE}  |||  MLENB: {correlation_coefficient_MLENB}  |||  TLE: {correlation_coefficient_TLE}\n")
        f.write(f"node;in_degree;local_intrinsic_dimensionality_LPCA;local_intrinsic_dimensionality_MLE;local_intrinsic_dimensionality_MLENB;local_intrinsic_dimensionality_TLE\n")
        for node, (in_degree, lidLPCA, lidMLE, lidsMLENB, lidTLE) in node_data.items():
            f.write(f"{node};{in_degree};{lidLPCA};{lidMLE};{lidsMLENB};{lidTLE}\n")
        f.close()
        return correlation_coefficient_LPCA,correlation_coefficient_MLE,correlation_coefficient_MLENB,correlation_coefficient_TLE,average_lids,skewness
    else:
        return False
