import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df_climate = pd.read_csv('climate_data.csv')

# Rename columns
df_climate = df_climate.rename(columns={
    "Bureau of Meteorology station number": "station_no",
    "Maximum temperature (Degree C)": "max",
    "Minimum temperature (Degree C)": "min",
    "Rainfall amount (millimetres)": "rainfall"
})

df_climate = df_climate.dropna()

# Combine Year, Month, Day into a proper datetime column
df_climate["Date"] = pd.to_datetime(df_climate[["Year", "Month", "Day"]])

# Define start and end dates (only month/day matter)
start = (6, 9)   # June 9
end = (9, 22)    # September 22

# Filter using month and day
mask = (
    (df_climate["Date"].dt.month > start[0]) & (df_climate["Date"].dt.month < end[0]) |
    ((df_climate["Date"].dt.month == start[0]) & (df_climate["Date"].dt.day >= start[1])) |
    ((df_climate["Date"].dt.month == end[0]) & (df_climate["Date"].dt.day <= end[1]))
)

df_climate = df_climate[mask]

# Remove original Year Month Day
df_climate = df_climate.drop(columns=['Day','Month','Year'])


df_climate = df_climate.rename(columns={
    "station_no": "mountain",
})

station_to_mountain = {
    72161: "Selwyn",
    71032: "Thredbo",
    83024: "Mt. Buller",
    83084: "Falls Creek",
    83085: "Mt. Hotham",
    85291: "Mt. Baw Baw",
    71075: "Perisher"
}

df_climate['mountain'] = df_climate['mountain'].replace(station_to_mountain)


# ADD MOUNT STIRLING
rows_to_duplicate = df_climate[df_climate['mountain'] == "Mt. Buller"].copy()
rows_to_duplicate['mountain'] = "Mt. Stirling"
df_climate = pd.concat([df_climate, rows_to_duplicate], ignore_index=True)


# ADD CHARLOTTE PASS
mountains_to_average = ["Thredbo", "Perisher"]
df_subset = df_climate[df_climate['mountain'].isin(mountains_to_average)]
# Optional - Filter out if both either Thredbo or Perisher don't have a value 
# date_counts = df_subset.groupby('Date')['mountain'].nunique()
# dates_with_both = date_counts[date_counts == 2].index
# df_subset = df_subset[df_subset['Date'].isin(dates_with_both)]
# End Optional Filter
df_avg = df_subset.groupby('Date', as_index=False)[['max','min','rainfall']].mean()
df_avg['mountain'] = "Charlotte Pass"
df_climate = pd.concat([df_climate, df_avg], ignore_index=True)


# FILTERS
df_climate = df_climate[df_climate["mountain"].isin(["Charlotte Pass", "Thredbo", "Perisher"])]

# df_climate = df_climate[df_climate["mountain"].isin(["Thredbo"])]
# df_climate = df_climate[df_climate["Date"] > "2023-07-28"]

# SORTATION
df_climate = df_climate.sort_values(by="Date", ascending=False).head(50)

# PRINT THE TABLE
print(df_climate)
