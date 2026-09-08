import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/processed/station_observations.csv")
OUTPUT_FILE = Path("data/synthetic/anomaly_dataset.csv")

RANDOM_SEED = 42


def inject_spike(df, index, rng, event_id):
    variable = rng.choice(["temperature_c", "pressure_hpa", "humidity_pct"])

    if variable == "temperature_c":
        df.loc[index, variable] += rng.choice([-1, 1]) * rng.uniform(10, 20)

    elif variable == "pressure_hpa":
        df.loc[index, variable] += rng.choice([-1, 1]) * rng.uniform(10, 20)

    else:
        df.loc[index, variable] = np.clip(
            df.loc[index, variable] + rng.choice([-1, 1]) * rng.uniform(30, 50),
            0,
            100
        )

    df.loc[index, "anomaly_label"] = 1
    df.loc[index, "anomaly_type"] = "spike"
    df.loc[index, "event_id"] = event_id


def inject_noise(df, index, rng, event_id):
    df.loc[index, "temperature_c"] += rng.normal(0, 3)
    df.loc[index, "pressure_hpa"] += rng.normal(0, 3)
    df.loc[index, "humidity_pct"] = np.clip(
        df.loc[index, "humidity_pct"] + rng.normal(0, 8),
        0,
        100
    )

    df.loc[index, "anomaly_label"] = 1
    df.loc[index, "anomaly_type"] = "noise"
    df.loc[index, "event_id"] = event_id


def inject_frozen(df, start, length, event_id):
    columns = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    end = min(start + length, len(df))

    for column in columns:
        value = df.loc[start, column]

        if pd.notna(value):
            df.loc[start:end - 1, column] = value

    df.loc[start:end - 1, "anomaly_label"] = 1
    df.loc[start:end - 1, "anomaly_type"] = "frozen"
    df.loc[start:end - 1, "event_id"] = event_id


def inject_drift(df, start, length, rng, event_id):
    end = min(start + length, len(df))
    actual_length = end - start

    if actual_length <= 1:
        return

    drift_temperature = rng.choice([-1, 1]) * rng.uniform(5, 12)
    drift_pressure = rng.choice([-1, 1]) * rng.uniform(5, 12)
    drift_humidity = rng.choice([-1, 1]) * rng.uniform(15, 30)

    for offset, index in enumerate(range(start, end)):
        progress = offset / (actual_length - 1)

        if pd.notna(df.loc[index, "temperature_c"]):
            df.loc[index, "temperature_c"] += drift_temperature * progress

        if pd.notna(df.loc[index, "pressure_hpa"]):
            df.loc[index, "pressure_hpa"] += drift_pressure * progress

        if pd.notna(df.loc[index, "humidity_pct"]):
            df.loc[index, "humidity_pct"] = np.clip(
                df.loc[index, "humidity_pct"] + drift_humidity * progress,
                0,
                100
            )

    df.loc[start:end - 1, "anomaly_label"] = 1
    df.loc[start:end - 1, "anomaly_type"] = "drift"
    df.loc[start:end - 1, "event_id"] = event_id


def inject_missing(df, start, length, event_id):
    end = min(start + length, len(df))

    columns = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    df.loc[start:end - 1, columns] = np.nan

    df.loc[start:end - 1, "anomaly_label"] = 1
    df.loc[start:end - 1, "anomaly_type"] = "missing"
    df.loc[start:end - 1, "event_id"] = event_id


def inject_cross_sensor(df, index, rng, event_id):
    variable = rng.choice([
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ])

    if variable == "temperature_c":
        df.loc[index, variable] += rng.uniform(8, 15)

    elif variable == "pressure_hpa":
        df.loc[index, variable] += rng.uniform(8, 15)

    else:
        df.loc[index, variable] = np.clip(
            df.loc[index, variable] + rng.uniform(25, 40),
            0,
            100
        )

    df.loc[index, "anomaly_label"] = 1
    df.loc[index, "anomaly_type"] = "cross_sensor"
    df.loc[index, "event_id"] = event_id


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["anomaly_label"] = 0
    df["anomaly_type"] = "normal"
    df["event_id"] = -1

    valid_indices = np.arange(len(df))

    used_indices = set()

    event_id = 1

    # Point anomalies
    point_anomalies = {
        "spike": 80,
        "noise": 60,
        "cross_sensor": 60
    }

    for anomaly_type, count in point_anomalies.items():

        candidates = [
            index for index in valid_indices
            if index not in used_indices
        ]

        selected = rng.choice(
            candidates,
            size=min(count, len(candidates)),
            replace=False
        )

        for index in selected:

            if anomaly_type == "spike":
                inject_spike(df, index, rng, event_id)

            elif anomaly_type == "noise":
                inject_noise(df, index, rng, event_id)

            elif anomaly_type == "cross_sensor":
                inject_cross_sensor(df, index, rng, event_id)

            used_indices.add(index)
            event_id += 1

    # Window anomalies
    window_anomalies = {
        "frozen": 8,
        "drift": 6,
        "missing": 6
    }

    for anomaly_type, count in window_anomalies.items():

        for _ in range(count):

            possible_start = [
                index
                for index in range(len(df) - 48)
                if all(
                    index + offset not in used_indices
                    for offset in range(48)
                )
            ]

            if not possible_start:
                break

            start = int(rng.choice(possible_start))

            if anomaly_type == "frozen":
                length = int(rng.integers(6, 24))
                inject_frozen(df, start, length, event_id)

            elif anomaly_type == "drift":
                length = int(rng.integers(12, 48))
                inject_drift(df, start, length, rng, event_id)

            else:
                length = int(rng.integers(2, 7))
                inject_missing(df, start, length, event_id)

            for offset in range(length):
                used_indices.add(start + offset)

            event_id += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Anomaly injection completed.")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    print("Anomaly distribution:")
    print(df["anomaly_type"].value_counts())

    print()
    print("Total anomalous rows:")
    print(int(df["anomaly_label"].sum()))


if __name__ == "__main__":
    main()