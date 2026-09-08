import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/station_observations.csv")
OUTPUT_FILE = Path("data/processed/validation_report.json")


def main():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    report = {}

    report["rows"] = len(df)
    report["stations"] = df["station_id"].nunique()

    report["time_start"] = str(df["timestamp"].min())
    report["time_end"] = str(df["timestamp"].max())

    report["duplicate_station_timestamps"] = int(
        df.duplicated(["station_id", "timestamp"]).sum()
    )

    report["missing_values"] = {
        column: int(df[column].isna().sum())
        for column in [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct"
        ]
    }

    report["invalid_ranges"] = {
        "temperature": int(
            ((df["temperature_c"] < -90) |
             (df["temperature_c"] > 60)).sum()
        ),
        "pressure": int(
            ((df["pressure_hpa"] < 800) |
             (df["pressure_hpa"] > 1100)).sum()
        ),
        "humidity": int(
            ((df["humidity_pct"] < 0) |
             (df["humidity_pct"] > 100)).sum()
        )
    }

    df = df.sort_values(["station_id", "timestamp"])

    gaps = df.groupby("station_id")["timestamp"].diff()

    report["timestamp_gap_counts"] = {
        str(k): int(v)
        for k, v in gaps.value_counts().items()
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("Validation completed.")
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()