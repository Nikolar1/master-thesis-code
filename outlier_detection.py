import datetime

import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils import calculate_hubness_normalized, find_outliers_knn, calculate_lid_weights
import dataset_analysis


def sort_by_score(scores):
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices

def calculate_scores(hubness_results, lid_results):
    return lid_results - 0.5 * hubness_results

def convert_labels(labels):
    return [1 if label == "yes" else 0 for label in labels]

def precission_at_n(indcies, labels, n):
    first_n_indicies = indcies[:n]
    count = 0
    for index in first_n_indicies:
        if labels[index] == 1:
            count += 1
    return count / n

if __name__ == '__main__':
    k_max = 101
    k_min = 40
    increments = 5
    n_jobs=12
    scores_for_dataset = {}
    metrics_for_dataset = {}
    datasets = [
        "Wilt_withoutdupl_norm_02_v10_outliers",
        "Stamps_withoutdupl_norm_05_v10_outliers",
        "SpamBase_withoutdupl_norm_20_v10_outliers",
        "Pima_withoutdupl_norm_20_v10_outlier",
        "Parkinson_withoutdupl_norm_20_v10_outliers",
        "PageBlocks_withoutdupl_norm_05_v10_outliers",
        "Hepatitis_withoutdupl_norm_10_v10_outliers",
        "HeartDisease_withoutdupl_norm_20_v10_outliers",
        "Cardiotocography_withoutdupl_norm_20_v10_outliers",
        "Arrhythmia_withoutdupl_norm_20_v10_outliers",
        "Annthyroid_05_v10_outlier",
        "glass_outlier",
        "Waveform_withoutdupl_norm_v10_outlier",
        "WBC_withoutdupl_norm_v10_outlier",
        "WDBC_withoutdupl_norm_v10_outlier",
        "WPBC_withoutdupl_norm_outlier",
        "Shuttle_withoutdupl_norm_v10_outlier",
        "KDDCup99_withoutdupl_norm_catremoved_outlier",
        "PenDigits_withoutdupl_norm_v10_outlier",
        "ALOI_withoutdupl_norm_outlier",
        "Ionosphere_withoutdupl_norm_outlier",
    ]
    for dataset_name in datasets:
        labels, data = dataset_analysis.load_dataset(dataset_name)
        if len(labels) < k_min:
            continue
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        k_values = range(min(k_min, len(labels)), min(k_max, len(labels)), increments)
        binary_labels = convert_labels(labels)
        dataset_values = {}
        dataset_metrics = {}
        for k in k_values:
            print(f"{datetime.datetime.now()}: {dataset_name} calculating {k}, of {max(k_values)}")
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=12)
            knn.fit(data, labels)
            lof = LocalOutlierFactor(n_neighbors=k, n_jobs=12).fit(data)
            lof_scores = -lof.negative_outlier_factor_
            knn_scores = find_outliers_knn(data, n_jobs=12, n_neighbors=k)

            hubness_results = calculate_hubness_normalized(data, labels, pre_trained_classifier=None, n_neighbors=min(len(labels) - 1, k * 10), n_jobs=12)
            lid_results = calculate_lid_weights(data, labels, pre_trained_classifier=knn, n_neighbors=k, n_jobs=12)
            dataset_values.update({k: (hubness_results.tolist(),lid_results.tolist())})

            combined_scores = calculate_scores(hubness_results, lid_results)


            roc_auc_val = roc_auc_score(binary_labels, combined_scores)
            average_precision = average_precision_score(binary_labels, combined_scores)
            precission_at_n_val = precission_at_n(sort_by_score(combined_scores), binary_labels, binary_labels.count(1))

            lof_roc_auc_val = roc_auc_score(binary_labels, lof_scores)
            lof_average_precision = average_precision_score(binary_labels, lof_scores)
            lof_precission_at_n_val = precission_at_n(sort_by_score(lof_scores), binary_labels, binary_labels.count(1))

            knn_roc_auc_val = roc_auc_score(binary_labels, knn_scores)
            knn_average_precision = average_precision_score(binary_labels, knn_scores)
            knn_precission_at_n_val = precission_at_n(sort_by_score(knn_scores), binary_labels, binary_labels.count(1))

            dataset_metrics.update({
                k:
                    (
                        roc_auc_val,average_precision,precission_at_n_val,
                        lof_roc_auc_val, lof_average_precision, lof_precission_at_n_val,
                        knn_roc_auc_val, knn_average_precision, knn_precission_at_n_val
                    )
            })

        scores_for_dataset.update({dataset_name: dataset_values})
        metrics_for_dataset.update({dataset_name: (max(k_values), dataset_metrics)})
        # with open(f'json_scores/{dataset_name}.json', 'w+') as f:
        #     json.dump(scores_for_dataset, f)

    k_values = range(k_min, k_max, increments)
    with open(f'outlier_data/results.csv','w+') as f:
        header = "dataset_name"
        for k in k_values:
            header += f";roc_{k};ap_{k};p@{k};lof_roc_{k};lof_ap_{k};lof_p@{k};knn_roc_{k};knn_ap_{k};knn_p@{k}"
        f.write(header + '\n')
        for dataset_name, (max_k, metrics_per_dataset) in metrics_for_dataset.items():
            roc_auc_vals = []
            average_precision_vals = []
            precission_at_n_vals = []
            lof_roc_auc_vals = []
            lof_average_precision_vals = []
            lof_precission_at_n_vals = []
            knn_roc_auc_vals = []
            knn_average_precision_vals = []
            knn_precission_at_n_vals = []
            k_values = range(k_min, max_k+1, increments)

            row = dataset_name
            for k, (roc_auc_val,average_precision,precission_at_n_val, lof_roc_auc_val, lof_average_precision, lof_precission_at_n_val,knn_roc_auc_val, knn_average_precision, knn_precission_at_n_val) in metrics_per_dataset.items():
                row+=f";{roc_auc_val};{average_precision};{precission_at_n_val};{lof_roc_auc_val};{lof_average_precision};{lof_precission_at_n_val};{knn_roc_auc_val};{knn_average_precision};{knn_precission_at_n_val}"
                roc_auc_vals.append(roc_auc_val)
                average_precision_vals.append(average_precision)
                precission_at_n_vals.append(precission_at_n_val)
                lof_roc_auc_vals.append(lof_roc_auc_val)
                lof_average_precision_vals.append(lof_average_precision)
                lof_precission_at_n_vals.append(lof_precission_at_n_val)
                knn_roc_auc_vals.append(knn_roc_auc_val)
                knn_average_precision_vals.append(knn_average_precision)
                knn_precission_at_n_vals.append(knn_precission_at_n_val)
            f.write(row + '\n')

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].plot(k_values, roc_auc_vals, label="LID-HUB ROC AUC")
            axes[0].plot(k_values, lof_roc_auc_vals, label="LOF ROC AUC")
            axes[0].plot(k_values, knn_roc_auc_vals, label="KNN ROC AUC")
            axes[0].set_xlabel('k')
            axes[0].set_ylabel('ROC AUC')
            axes[0].set_title('ROC AUC')
            axes[0].legend()

            axes[1].plot(k_values, average_precision_vals, label="LID-HUB Average Precision")
            axes[1].plot(k_values, lof_average_precision_vals, label="LOF Average Precision")
            axes[1].plot(k_values, knn_average_precision_vals, label="KNN Average Precision")
            axes[1].set_xlabel('k')
            axes[1].set_ylabel('Average Precision')
            axes[1].set_title('Average Precision')
            axes[1].legend()

            axes[2].plot(k_values, precission_at_n_vals, label="LID-HUB Precision at N")
            axes[2].plot(k_values, lof_precission_at_n_vals, label="LOF Precision at N")
            axes[2].plot(k_values, knn_precission_at_n_vals, label="KNN Precision at N")
            axes[2].set_xlabel('k')
            axes[2].set_ylabel('Precision at N')
            axes[2].set_title('Precision at N')
            axes[2].legend()

            # Show the plots
            plt.tight_layout()
            plt.savefig(f'outlier_data/{dataset_name}.png')



    # binary_labels = convert_labels(labels)
    # roc_auc_val = roc_auc_score(binary_labels, calculate_scores(hubness_results, lid_results))
    # average_precision = average_precision_score(binary_labels, calculate_scores(hubness_results, lid_results))
    # print(f"ROC AUC: {roc_auc_val}")
    # print(f"Average precission: {average_precision}")
