"""Create the flat, analysis ready source used by the Tableau dashboard."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outage.csv"
OUTPUT = Path(__file__).resolve().parent / "outage_tableau.csv"

KEEP = [
    "OBS",
    "YEAR",
    "MONTH",
    "U.S._STATE",
    "POSTAL.CODE",
    "NERC.REGION",
    "CLIMATE.REGION",
    "CLIMATE.CATEGORY",
    "OUTAGE.START.DATE",
    "OUTAGE.START.TIME",
    "OUTAGE.RESTORATION.DATE",
    "OUTAGE.RESTORATION.TIME",
    "CAUSE.CATEGORY",
    "CAUSE.CATEGORY.DETAIL",
    "OUTAGE.DURATION",
    "CUSTOMERS.AFFECTED",
    "POPDEN_URBAN",
]


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def duration_bucket(hours: float) -> str:
    if hours < 24:
        return "Under 1 day"
    if hours < 72:
        return "1 to 3 days"
    return "More than 3 days"


def main() -> None:
    raw = pd.read_csv(SOURCE, header=5, low_memory=False).iloc[1:].copy()
    data = raw[KEEP].copy()

    numeric = ["OBS", "YEAR", "MONTH", "OUTAGE.DURATION", "CUSTOMERS.AFFECTED", "POPDEN_URBAN"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    required = ["YEAR", "MONTH", "U.S._STATE", "CLIMATE.REGION", "CAUSE.CATEGORY", "OUTAGE.DURATION"]
    data = data.dropna(subset=required)
    data = data.loc[data["OUTAGE.DURATION"] > 0].copy()

    data["YEAR"] = data["YEAR"].astype(int)
    data["MONTH"] = data["MONTH"].astype(int)
    data["Event ID"] = data["OBS"].astype(int)
    data["State"] = data["U.S._STATE"].str.strip()
    data["Postal Code"] = data["POSTAL.CODE"].str.strip()
    data["NERC Region"] = data["NERC.REGION"].str.strip()
    data["Climate Region"] = data["CLIMATE.REGION"].str.strip()
    data["Climate Category"] = data["CLIMATE.CATEGORY"].str.strip().str.title()
    data["Cause Category"] = data["CAUSE.CATEGORY"].str.strip().str.title()
    data["Cause Detail"] = data["CAUSE.CATEGORY.DETAIL"].fillna("").str.strip().str.title()
    data["Duration Minutes"] = data["OUTAGE.DURATION"]
    data["Duration Hours"] = data["OUTAGE.DURATION"] / 60
    data["Customers Affected"] = data["CUSTOMERS.AFFECTED"]
    data["Urban Population Density"] = data["POPDEN_URBAN"]
    data["Season"] = data["MONTH"].map(season)
    data["Duration Bucket"] = data["Duration Hours"].map(duration_bucket)
    data["Outage Count"] = 1

    start_date = pd.to_datetime(data["OUTAGE.START.DATE"], errors="coerce")
    start_time = data["OUTAGE.START.TIME"].fillna("00:00:00").astype(str)
    data["Outage Start"] = pd.to_datetime(
        start_date.dt.strftime("%Y-%m-%d") + " " + start_time,
        format="mixed",
        errors="coerce",
    )

    output_columns = [
        "Event ID",
        "Outage Start",
        "YEAR",
        "MONTH",
        "Season",
        "State",
        "Postal Code",
        "Climate Region",
        "NERC Region",
        "Climate Category",
        "Cause Category",
        "Cause Detail",
        "Duration Minutes",
        "Duration Hours",
        "Duration Bucket",
        "Customers Affected",
        "Urban Population Density",
        "Outage Count",
    ]

    data = data[output_columns].sort_values(["Outage Start", "Event ID"])
    data.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"Wrote {len(data):,} rows to {OUTPUT}")
    print(f"Years: {data['YEAR'].min()} to {data['YEAR'].max()}")
    print(f"Cause categories: {data['Cause Category'].nunique()}")


if __name__ == "__main__":
    main()
