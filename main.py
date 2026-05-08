"""Top-level entrypoint for training and prediction.

Usage:
  python main.py train --target target --test-size 0.2
  python main.py predict --input some_file.csv
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML project CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train model (sample demo data)")
    train_p.add_argument("--target", default="target", help="Target column name")
    train_p.add_argument("--test-size", type=float, default=0.2, help="Test size")
    train_p.add_argument(
        "--model-type",
        default="regression",
        choices=["regression", "classification"],
        help="Model type",
    )

    pred_p = sub.add_parser("predict", help="Predict using trained model")
    pred_p.add_argument("--input", required=True, help="Path to CSV file")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        # Import inside to keep CLI light.
        from src.pipeline.train_pipeline import TrainPipeline
        import pandas as pd
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df[args.target] = y

        pipeline = TrainPipeline()
        pipeline.run_pipeline(
            dataframe=df,
            target_column=args.target,
            test_size=args.test_size,
            model_type=args.model_type,
        )

    elif args.command == "predict":
        from src.pipeline.predict_pipeline import PredictionPipeline

        pipeline = PredictionPipeline()
        preds = pipeline.batch_predict(args.input)
        # Print a preview for convenience
        try:
            print(preds[:10])
        except Exception:
            print(preds)


if __name__ == "__main__":
    main()

