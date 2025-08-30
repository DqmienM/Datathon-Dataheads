import pandas as pd
import matplotlib.pyplot as plt
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

# Sort ascending (lowest first)
# df_climate = df_climate.sort_values(by="min").head(20)

df_climate = df_climate.sort_values(by="Date", ascending=False).head(20)

df_climate = df_climate.sort_values(by="Date", ascending=False).head(20)

df_climate = sf_filtered[sf_filtered["Bureau of Meteorology station number"] == 1]

# df_climate = df_climate.sort_values(by="min").head(20)

# df_climate = df_climate.sort_values(by="min").head(20)

print(df_climate.to_string())

plt.figure(figsize=(12,6))
plt.plot(df_climate['Date'], df_climate['max'], df_climate['71075'], label='Max Temp')
plt.plot(df_climate['Date'], df_climate['min'], df_climate['71075'], label='Min Temp')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperatures Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# df_climate["Year"].value_counts()