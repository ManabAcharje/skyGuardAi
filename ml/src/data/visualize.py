import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/processed/station_observations.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df["temperature_c"])
    plt.title("Temperature over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df["pressure_hpa"])
    plt.title("Pressure over Time")
    plt.xlabel("Time")
    plt.ylabel("Pressure (hPa)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df["humidity_pct"])
    plt.title("Relative Humidity over Time")
    plt.xlabel("Time")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()