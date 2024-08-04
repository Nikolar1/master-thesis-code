from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from dataset_analysis import load_dataset
import numpy as np
from pathlib import Path


def find_best_k(dataset_name: str, dataset_percentage = None, k_range = None, n_jobs: int = 1, n_splits=5):
    if k_range:
        k_values = k_range
    else:
        k_values = [i for i in range(3, 31)]

    loaded_dataset = load_dataset(dataset_name, dataset_percentage)
    if loaded_dataset:
        labels, data = loaded_dataset
        scores_dict = {}
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            score = cross_val_score(knn, data, labels, cv=skf)
            mean_score = np.mean(score)
            scores_dict.update({k: (mean_score, score)})
        Path("./best_k").mkdir(exist_ok=True)
        f = open(f"./best_k/{dataset_name}.txt", "w+")
        header = "k;mean_score;number_of_scores;"
        for i in range(1, n_splits+1):
            header += f"score_{i}"
        f.write(header + "\n")
        for k, (mean_score, scores) in scores_dict.items():
            str_to_write = f"{k};{mean_score};{np.size(scores)};"
            for score in scores:
                str_to_write += f"{score};"
            str_to_write += "\n"
            f.write(str_to_write)
        f.close()
        max_k = max(scores_dict, key=lambda k: scores_dict[k][0])
        return max_k
    else:
        return False
