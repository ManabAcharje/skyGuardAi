import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


INPUT_FILE = Path("data/processed/features.csv")
MODEL_FILE = Path("ml/models/isolation_forest.joblib")

TRAIN_RATIO = 0.70


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # SAME TIME SPLIT AS TRAINING
    # --------------------------------------------------

    split_index = int(len(df) * TRAIN_RATIO)

    test_df = df.iloc[split_index:].copy()

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    imputer = artifact["imputer"]
    feature_columns = artifact["feature_columns"]

    X_test = test_df[feature_columns]

    X_test = imputer.transform(X_test)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    raw_predictions = model.predict(X_test)

    predictions = np.where(
        raw_predictions == -1,
        1,
        0
    )

    y_true = test_df["anomaly_label"].values

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    print("=" * 60)
    print("SKYGUARD AI - ISOLATION FOREST EVALUATION")
    print("=" * 60)
    print()

    print("Test rows:", len(test_df))
    print("Actual anomalies:", int(y_true.sum()))
    print("Predicted anomalies:", int(predictions.sum()))
    print()

    print("CONFUSION MATRIX")
    print("-" * 30)

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")
    print()

    print("METRICS")
    print("-" * 30)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()

    print("CLASSIFICATION REPORT")
    print("-" * 30)

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Normal",
                "Anomaly"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # PERFORMANCE BY ANOMALY TYPE
    # --------------------------------------------------

    test_df["prediction"] = predictions

    anomaly_types = [
        "spike",
        "noise",
        "frozen",
        "drift",
        "cross_sensor",
        "missing"
    ]

    print("ANOMALY TYPE PERFORMANCE")
    print("-" * 30)

    for anomaly_type in anomaly_types:

        subset = test_df[
            test_df["anomaly_type"] == anomaly_type
        ]

        if len(subset) == 0:
            continue

        actual = subset["anomaly_label"].values
        predicted = subset["prediction"].values

        detected = int(
            ((actual == 1) & (predicted == 1)).sum()
        )

        total = int(
            (actual == 1).sum()
        )

        detection_rate = (
            detected / total
            if total > 0
            else 0
        )

        print(
            f"{anomaly_type:15s} "
            f"{detected:3d}/{total:3d} "
            f"({detection_rate:.2%})"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()