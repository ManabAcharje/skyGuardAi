import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/isd_parsed.csv")
OUTPUT_FILE = Path("data/processed/station_observations.csv")

#August-Roche-Magnus formula-> to calculate relative humidity from temperature and dew point
def calculate_relative_humidity(temperature, dew_point):
    a = 17.625
    b = 243.04

    rh = 100 * np.exp(
        (a * dew_point / (b + dew_point))
        - (a * temperature / (b + temperature))
    )

    return np.clip(rh, 0, 100)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    valid = (
        df["temperature_c"].notna()
        & df["dew_point_c"].notna()
    )

    df["humidity_pct"] = np.nan

    df.loc[valid, "humidity_pct"] = calculate_relative_humidity(
        df.loc[valid, "temperature_c"],
        df.loc[valid, "dew_point_c"]
    )

    result = df[
        [
            "station_id",
            "timestamp",
            "temperature_c",
            "pressure_hpa",
            "humidity_pct"
        ]
    ]

    result = result.sort_values("timestamp")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("Normalization completed.")
    print(f"Rows: {len(result)}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print(result.head())


if __name__ == "__main__":
    main()