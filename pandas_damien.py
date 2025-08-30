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

# Sort ascending (lowest first)
# df_climate = df_climate.sort_values(by="min").head(20)



df_climate = df_climate[df_climate["station_no"] == 71075]

df_climate = df_climate.sort_values(by="Date", ascending=False)

# df_climate = df_climate.sort_values(by="Date", ascending=False).head(20)



# df_climate = df_climate.sort_values(by="min").head(20)

# df_climate = df_climate.sort_values(by="min").head(20)



plt.figure(figsize=(18,6))
plt.plot(df_climate['Date'], df_climate['max'], label='Max Temp')
plt.plot(df_climate['Date'], df_climate['min'], label='Min Temp')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))   # every 1 year

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperatures Over Time For Station = 71075')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# df_climate["Year"].value_counts()