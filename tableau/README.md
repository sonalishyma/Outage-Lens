# Tableau dashboard source

This folder contains the reproducible flat data source for the Outage Lens Tableau dashboard.

Run:

```bash
python tableau/prepare_tableau_data.py
```

The script reads the original Purdue LASCI `outage.csv`, removes the units row, coerces numeric fields, excludes zero or missing duration records, standardizes labels, and derives duration hours, season, duration bucket, and outage start. The output contains 1,393 analysis ready outage records from 2000 through 2016.

The intended dashboard uses three views:

1. Median duration by recorded cause category
2. Outage count over time
3. State level outage duration and frequency

The dashboard should be described as Tableau work only after a published Tableau Public workbook exists.

