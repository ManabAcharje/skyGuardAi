import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/synthetic/anomaly_dataset.csv")
OUTPUT_FILE = Path("data/processed/features.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    sensor_columns = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    # Missing-value indicators
    for column in sensor_columns:
        df[f"{column}_missing"] = df[column].isna().astype(int)

    # Temporal features
    hour = df["timestamp"].dt.hour
    day_of_year = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # Lag and difference features
    for column in sensor_columns:

        df[f"{column}_diff_1h"] = (
            df.groupby("station_id")[column].diff(1)
        )

        df[f"{column}_diff_3h"] = (
            df.groupby("station_id")[column].diff(3)
        )

        df[f"{column}_diff_6h"] = (
            df.groupby("station_id")[column].diff(6)
        )

        # Rolling statistics
        grouped = df.groupby("station_id")[column]

        df[f"{column}_rolling_mean_6h"] = (
            grouped.transform(
                lambda x: x.rolling(6, min_periods=3).mean()
            )
        )

        df[f"{column}_rolling_std_6h"] = (
            grouped.transform(
                lambda x: x.rolling(6, min_periods=3).std()
            )
        )

        df[f"{column}_rolling_mean_24h"] = (
            grouped.transform(
                lambda x: x.rolling(24, min_periods=6).mean()
            )
        )

        df[f"{column}_rolling_std_24h"] = (
            grouped.transform(
                lambda x: x.rolling(24, min_periods=6).std()
            )
        )

    # Rate-of-change features
    df["temperature_rate"] = df["temperature_c_diff_1h"]
    df["pressure_rate"] = df["pressure_hpa_diff_1h"]
    df["humidity_rate"] = df["humidity_pct_diff_1h"]

    # Cross-sensor relationship
    df["temperature_humidity_interaction"] = (
        df["temperature_c"] * df["humidity_pct"]
    )

    df["pressure_temperature_ratio"] = (
        df["pressure_hpa"] / df["temperature_c"].abs().clip(lower=1)
    )

    # Keep labels for evaluation
    result_columns = [
        "station_id",
        "timestamp",

        "temperature_c",
        "pressure_hpa",
        "humidity_pct",

        "temperature_c_missing",
        "pressure_hpa_missing",
        "humidity_pct_missing",

        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",

        "temperature_c_diff_1h",
        "temperature_c_diff_3h",
        "temperature_c_diff_6h",

        "pressure_hpa_diff_1h",
        "pressure_hpa_diff_3h",
        "pressure_hpa_diff_6h",

        "humidity_pct_diff_1h",
        "humidity_pct_diff_3h",
        "humidity_pct_diff_6h",

        "temperature_c_rolling_mean_6h",
        "temperature_c_rolling_std_6h",
        "temperature_c_rolling_mean_24h",
        "temperature_c_rolling_std_24h",

        "pressure_hpa_rolling_mean_6h",
        "pressure_hpa_rolling_std_6h",
        "pressure_hpa_rolling_mean_24h",
        "pressure_hpa_rolling_std_24h",

        "humidity_pct_rolling_mean_6h",
        "humidity_pct_rolling_std_6h",
        "humidity_pct_rolling_mean_24h",
        "humidity_pct_rolling_std_24h",

        "temperature_rate",
        "pressure_rate",
        "humidity_rate",

        "temperature_humidity_interaction",
        "pressure_temperature_ratio",

        "anomaly_label",
        "anomaly_type",
        "event_id"
    ]

    result = df[result_columns]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("Feature engineering completed.")
    print(f"Rows: {len(result)}")
    print(f"Features: {len(result.columns)}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    print("Missing values:")
    print(result.isna().sum())


if __name__ == "__main__":
    main()