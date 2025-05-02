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
        "o2_comBat": {"model": "rf", "accuracy": 0.66},
        "po4_comBat": {"model": "knn", "accuracy": 0.59},
        "tp8_comBat": {"model": "rf", "accuracy": 0.7},
        "oz_comBat": {"model": "lda", "accuracy": 0.65},
    }

    # Save configuration to JSON file
    with open("channels_config.json", "w") as f:
        json.dump(channels_models, f)

    # Create and run the trainer
    trainer = SupremeTrainer(
        channels_models=channels_models,
        cv_method=args.cv_method,
        top_n_labels=2,
        n_features_to_select=15,
        lmoso_leftout=2,
        permu_count=100,
    )

    # Run complete analysis
    trainer.run_complete_analysis(n_iter=5)


if __name__ == "__main__":
    main()
