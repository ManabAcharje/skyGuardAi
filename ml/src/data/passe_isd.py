import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/raw/noaa_isd/original.csv")
OUTPUT_FILE = Path("data/processed/isd_parsed.csv")


def parse_value(value):
    if pd.isna(value):
        return None

    value = str(value).split(",")[0]

    if value in {"+9999", "99999", "-9999"}:
        return None

    return float(value)


def main():
    df = pd.read_csv(INPUT_FILE)

    result = pd.DataFrame()

    result["station_id"] = df["STATION"]
    result["timestamp"] = pd.to_datetime(df["DATE"])

    result["temperature_c"] = df["TMP"].apply(parse_value) / 10
    result["dew_point_c"] = df["DEW"].apply(parse_value) / 10
    result["pressure_hpa"] = df["SLP"].apply(parse_value) / 10

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("Parsing completed.")
    print(f"Rows: {len(result)}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print(result.head())


if __name__ == "__main__":
    main()