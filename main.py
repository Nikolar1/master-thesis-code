from dataset_analysis import analyze_dataset
from datasets import datasets
from find_best_k import find_best_k
import logging.config

DATASET_PERCENTAGE_FOR_BEST_K = None
DATASET_PERCENTAGE_FOR_ANALYSIS = None

if __name__ == '__main__':
    logging.config.fileConfig('logger.conf')
    logger = logging.getLogger('Main')
    logger.info("----------------------     Starting main.py       ----------------------------")
    logger.info(f"Current run settings: DATASET_PERCENTAGE_FOR_BEST_K={DATASET_PERCENTAGE_FOR_BEST_K}, DATASET_PERCENTAGE_FOR_ANALYSIS={DATASET_PERCENTAGE_FOR_ANALYSIS}")
    for dataset_title, dataset_name in datasets.items():
        logger.info(f"Finding best k for dataset '{dataset_title}'")
        k = find_best_k(dataset_name, dataset_percentage=DATASET_PERCENTAGE_FOR_BEST_K, n_jobs=12)
        if k:
            logger.info(f"Best k for dataset '{dataset_title}' is {k}, analyzing dataset...")
            result = analyze_dataset(dataset_name=dataset_name, k=k, n_jobs=12, dataset_percentage=DATASET_PERCENTAGE_FOR_ANALYSIS)
        else:
            logger.info(f"Failed to find best k for dataset '{dataset_title}' using k=3, analyzing dataset...")
            result = analyze_dataset(dataset_name=dataset_name, k=3, n_jobs=12, dataset_percentage=DATASET_PERCENTAGE_FOR_ANALYSIS)
        if result:
            logger.info(f"Successfully analyzed dataset '{dataset_title}', Correlation Coefficient: {result}")
        else:
            logger.info(f"Failed to analyze dataset '{dataset_title}'")
