import joblib
import argparse

def print_model_info(model):
    print("---- MODEL INFO ----\n")

    if hasattr(model, "steps"):  # It's a Pipeline
        print("This is a Pipeline:")
        for name, step in model.steps:
            print(f"  Step: {name} → {type(step).__name__}")
            if hasattr(step, 'get_params'):
                print("    Parameters:")
                for param, val in step.get_params().items():
                    print(f"      {param}: {val}")
            print()
    else:  # It's a plain model
        print(f"Model type: {type(model).__name__}")
        print("Parameters:")
        for param, val in model.get_params().items():
            print(f"  {param}: {val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a saved sklearn model (.pkl)")
    parser.add_argument("--model_path", type=str, default='ml_model_outputs/63%_svm/best_randomforest_model.pkl',
                        help="Path to the .pkl model file")

    args = parser.parse_args()
    model = joblib.load(args.model_path)
    print_model_info(model)
