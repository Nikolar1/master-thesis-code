from dataset_analysis import analyze_dataset
from datasets import datasets
from find_best_k import find_best_k
import logging.config
import numpy as np

DATASET_PERCENTAGE_FOR_BEST_K = None
DATASET_PERCENTAGE_FOR_ANALYSIS = None
N_JOBS = 6

if __name__ == '__main__':
    logging.config.fileConfig('logger.conf')
    logger = logging.getLogger('Main')
    logger.info("----------------------     Starting main.py       ----------------------------")
    logger.info(f"Current run settings: DATASET_PERCENTAGE_FOR_BEST_K={DATASET_PERCENTAGE_FOR_BEST_K}, DATASET_PERCENTAGE_FOR_ANALYSIS={DATASET_PERCENTAGE_FOR_ANALYSIS}")
    dataset_stats = {}
    for k in [20, 40, 60, 80, 100]:
        dataset_stats.update({k: {}})
    for dataset_title, dataset_name in datasets.items():
        logger.info(f"Working on dataset '{dataset_title}'")
        for k in [20, 40, 60, 80, 100]:
            if k:
                logger.info(f"Dataset '{dataset_title}' with {k}, analyzing dataset...")
                result = analyze_dataset(dataset_name=dataset_name, k=k, n_jobs=N_JOBS, dataset_percentage=DATASET_PERCENTAGE_FOR_ANALYSIS)
            else:
                logger.info(f"k missing for dataset '{dataset_title}' using k=3, analyzing dataset...")
                result = analyze_dataset(dataset_name=dataset_name, k=3, n_jobs=N_JOBS, dataset_percentage=DATASET_PERCENTAGE_FOR_ANALYSIS)
            if result:
                logger.info(f"Successfully analyzed dataset '{dataset_title}', Correlation Coefficients: LPCA:{result[0]}, MLE:{result[1]}, MLENB:{result[2]}, TLE:{result[3]}")
                dataset_stats[k].update({dataset_name: (result[4], result[5])})
            else:
                logger.info(f"Failed to analyze dataset '{dataset_title}'")
    for k, dataset_stats_for_k in dataset_stats.items():
        lidsLPCA = []
        lidsMLE = []
        lidsMLENB = []
        lidsTLE = []
        skewnesses = []
        for dataset_name, ((lidLPCA,lidMLE,lidMLENB,lidTLE), skewness) in dataset_stats_for_k.items():
            lidsLPCA.append(lidLPCA)
            lidsMLE.append(lidMLE)
            lidsMLENB.append(lidMLENB)
            lidsTLE.append(lidTLE)
            skewnesses.append(skewness)
        correlation_coefficient_LPCA = np.corrcoef(skewnesses, lidsLPCA)[0, 1]
        correlation_coefficient_MLE = np.corrcoef(skewnesses, lidsMLE)[0, 1]
        correlation_coefficient_MLENB = np.corrcoef(skewnesses, lidsMLENB)[0, 1]
        correlation_coefficient_TLE = np.corrcoef(skewnesses, lidsTLE)[0, 1]
        logger.info(f"Correlation coefficients for k={k}: LPCA: {correlation_coefficient_LPCA}  |||  MLE: {correlation_coefficient_MLE}  |||  MLENB: {correlation_coefficient_MLENB}  |||  TLE: {correlation_coefficient_TLE}")
