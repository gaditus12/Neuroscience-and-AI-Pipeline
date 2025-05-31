# example_run.py
import json
import argparse
from supreme_trainer import SupremeTrainer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Supreme EEG Model Trainer")
    parser.add_argument(
        "--cv_method",
        type=str,
        default="loso",
        choices=["loso", "lmoso", "kfold"],
        help="Cross-validation method (default: loso)",
    )
    args = parser.parse_args()

    # Define channels, models, and accuracies
    channels_models = {
        #"po4_spi_norm-combat": {"model": "hgb", "accuracy": 0.63},
        # "o2_spi_norm-combat": {"model": "svm", "accuracy": 0.6},
         #"po4_spi": {"model": "lda", "accuracy": 0.61},
        "po4_spi_norm-z1": {"model": "svm", "accuracy": 0.6},
        #"po4_spi_norm-z2": {"model": "knn", "accuracy": 0.6},
        #"po4_spi_norm-z3": {"model": "rf", "accuracy": 0.6},
        #"po4_spi_norm-z4": {"model": "hgb", "accuracy": 0.6},
        "o2_spi_norm-z": {"model": "lda", "accuracy": 0.6},
        #"po4_spi_norm-z6": {"model": "et", "accuracy": 0.6},
        #"po4_spi_norm-z7": {"model": "logreg", "accuracy": 0.6},
        #"po4_spi_norm-z8": {"model": "gnb", "accuracy": 0.6},
        "po4_spi1": {"model": "svm", "accuracy": 0.6},
        #"po4_spi2": {"model": "knn", "accuracy": 0.57},
        #"po4_spi3": {"model": "rf", "accuracy": 0.57},
        "o2_spi": {"model": "hgb", "accuracy": 0.6},
        "o2_spi_norm-comBat": {"model": "lda", "accuracy": 0.6},
        #"po4_spi6": {"model": "et", "accuracy": 0.57},
        #"po4_spi7": {"model": "logreg", "accuracy": 0.57},
        #"po4_spi8": {"model": "gnb", "accuracy": 0.57},
        # 'o2_spi': {"model": "svm", "accuracy": 0.6},
        # "po4_spi": {"model": "svm", "accuracy": 0.57},
    }

    # Save configuration to JSON file
    with open("channels_config.json", "w") as f:
        json.dump(channels_models, f)

    # Create and run the trainer
    trainer = SupremeTrainer(
        channels_models=channels_models,
        cv_method=args.cv_method,
        top_n_labels=2,
        n_features_to_select=5,
        lmoso_leftout=2,
        permu_count=1,
    )

    # Run complete analysis
    trainer.run_complete_analysis(n_iter=5)


if __name__ == "__main__":
    main()
