import argparse
import train
from train import train_model  # import train_model from train.py
import evaluate # import evaluate_model from evaluate.py
from evaluate import model_eval# import model_eval from evaluate.py

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate traffic sign classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    if args.train:
        train_model()

    if args.evaluate:
        model_eval()

if __name__ == "__main__":
    main()