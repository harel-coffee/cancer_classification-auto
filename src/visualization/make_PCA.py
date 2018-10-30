import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import src
import logging
import os


def main():
    
    output_path = src.reports_dir / "figures" / "PCA"
    os.makedirs(output_path, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    for cohort in src.data.TCGA_COHORTS:
        logger.info("Loading data of {} cohort".format(cohort))
        X, y, _, _, class_mapping = src.data.load_sample_classification_problem(cohort)
        if X.shape[0] > 0:
            logger.info("Plotting PCA of {} cohort".format(cohort))
            plt.figure(figsize=(10, 8))
            src.visualization.plot_PCA(X, y, class_mapping)
            plt.title(cohort, fontsize=14)
            plt.savefig(output_path / (cohort + ".png"))
            plt.close()
        else:
            logger.info("{} cohort has no samples...skipping".format(cohort))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()