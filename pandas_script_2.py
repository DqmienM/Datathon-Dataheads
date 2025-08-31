import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_cli = pd.read_csv('climate_data.csv')
#renam
df_cli = df_cli.rename(columns={
    "Bureau of Meteorology station number": "station_no",
    "Maximum temperature (Degree C)": "max",
    "Minimum temperature (Degree C)": "min",
    "Rainfall amount (millimetres)": "rainfall"
})
df_vis = pd.read_csv('visitation_data.csv')

# FILTER VISITATION DATA

mountains = ["Mt. Baw Baw","Mt. Stirling","Mt. Hotham","Falls Creek","Mt. Buller","Selwyn","Thredbo","Perisher","Charlotte Pass"]

#Filtering out weeks with 0 visitors for all mountains (covid)
df_vis = df_vis[(df_vis[mountains] != 0).any(axis=1)]

#filtering out null columns
df_vis = df_vis.dropna()

#converting values to int
df_vis["Year"] = df_vis["Year"].astype(int)
df_vis["Week"] = df_vis["Week"].astype(int)
for mount in mountains:
    df_vis[mount] = df_vis[mount].astype(int)

df_vis_weekly = df_vis #should work lmao


#create a datetime for the climate table
df_cli["date"] = pd.to_datetime(df_cli[["Year", "Month", "Day"]], errors="coerce")

# Example column
col = "rainfall"

# Compute z-scores
z = (df_cli[col] - df_cli[col].mean()) / df_cli[col].std()

# Keep rows within 3 standard deviations
df_cli = df_cli[np.abs(z) <= 3]


week1_start = pd.to_datetime( df_vis["Year"].astype(str) + "-05-09")
# week1_start = week1_start + pd.offsets.Week(weekday=0)
# df_vis["date"] = pd.to_datetime(df_vis[["Year"]], errors="coerce" )
df_vis["week_start"] = week1_start + pd.to_timedelta((df_vis["Week"] - 1) * 7, unit="D")
df_vis["dates"] = df_vis["week_start"].apply(lambda d: pd.date_range(d, periods=7, freq="D"))
df_vis = df_vis.explode("dates", ignore_index=True).rename(columns={"dates": "date"})
df_vis = df_vis.drop(columns=["Year","week_start"])




# Remove original Year Month Day
df_cli = df_cli.drop(columns=['Day','Month','Year'])


df_cli = df_cli.rename(columns={
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

df_cli['mountain'] = df_cli['mountain'].replace(station_to_mountain)


# ADD MOUNT STIRLING
rows_to_duplicate = df_cli[df_cli['mountain'] == "Mt. Buller"].copy()
rows_to_duplicate['mountain'] = "Mt. Stirling"
df_cli = pd.concat([df_cli, rows_to_duplicate], ignore_index=True)


# ADD CHARLOTTE PASS
mountains_to_average = ["Thredbo", "Perisher"]
df_subset = df_cli[df_cli['mountain'].isin(mountains_to_average)]
df_avg = df_subset.groupby('date', as_index=False)[['max','min','rainfall']].mean()
df_avg['mountain'] = "Charlotte Pass"
df_cli = pd.concat([df_cli, df_avg], ignore_index=True)

#weekly cli
#-----------------------

value_cols = ["max", "min", "rainfall"]

df_cli_weekly = df_cli

# ensure datetime
df_cli_weekly["date"] = pd.to_datetime(df_cli_weekly["date"], errors="coerce")

# extract ISO year/week
# iso = df_cli_weekly["date"].dt.isocalendar()              # returns a DataFrame: year, week, day
# df_cli_weekly["year"] = iso["year"].astype(int)
# df_cli_weekly["week"] = iso["week"].astype(int)

# # aggregate by ISO year/week
# # min_count=1 keeps NaN-only weeks as NaN; use min_count=len(value_cols) to require all present
# weekly = (
#     df.groupby(["year", "week"], as_index=False)[value_cols]
#       .mean()   # average each column
# )

# If you want the Monday of each ISO week too:
# weekly["week_start"] = pd.to_datetime(
#     weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2) + "-1",
#     format="%G-W%V-%u"
# )

#-----------------------

# print(df_cli)

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)
# df_vis.plot()[t.]
print("\n\n")

# print(df_vis.head(120))
# print(df_cli.head(120))

#Join is uniquely identified through station_no and date i thinkmien
df_join = pd.merge(df_vis, df_cli, on=["date"], how="inner")

#turn the corresponding mountain into a visitors
df_join["visitors"] = df_join.apply(lambda row: row[row["mountain"]]/7, axis=1)

df_join = df_join.drop(mountains, axis=1)
df_join = df_join.drop("Week", axis=1)

# print(df_join.columns.tolist())

# df_climate = df_climate.rename(columns={
#     "station_no": "mountain",
# })

df_join["avg_visitors_over_next_week"] = np.nan

# sort by mountain and date
df_join = df_join.sort_values(['mountain', 'date'])

# rolling average over current + next 6 days per mountain
df_join['avg_visitors_over_next_week'] = (
    df_join.groupby('mountain')['visitors']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)



df_join["avg_snowfall"] = np.nan

# sort by mountain and date
df_join = df_join.sort_values(['mountain', 'date'])

# rolling average over current + next 6 days per mountain
df_join['avg_snowfall'] = (
    df_join.groupby('mountain')['rainfall']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)


df_join['visitors'] = df_join['visitors'] * 7






# VISUALISATION

# Get a list of mountains
# mountains = df_join['mountain'].unique()

# plt.figure(figsize=(12,6))

# for mtn in mountains:
#     subset = df_join[df_join['mountain'] == mtn]
#     plt.plot(subset['date'], subset['visitors'], label=mtn)

# plt.xlabel('Date')
# plt.ylabel('Average Visitors Over Next Week')
# plt.title('Average Visitors per Mountain Over Time')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Filter to June 9 → Sep 22
# Filter to June 9 → Sep 22
# Filter to June 9 → September 22
# Filter to June 9 → Sep 22
# Filter June 9 → Sep 22
df_filtered = df_join[
    ((df_join['date'].dt.month > 6) | ((df_join['date'].dt.month == 6) & (df_join['date'].dt.day >= 9))) &
    ((df_join['date'].dt.month < 9) | ((df_join['date'].dt.month == 9) & (df_join['date'].dt.day <= 22)))
]

mountains = df_filtered['mountain'].unique()
years = sorted(df_filtered['date'].dt.year.unique())

plt.figure(figsize=(16,6))

# width of each year block
year_width = 80  # arbitrary, just to separate seasons visually
offset = 0

for year in years:
    season = df_filtered[df_filtered['date'].dt.year == year].copy()
    # normalize x-axis within the season: June 9 = 0
    season['day_in_season'] = (season['date'] - pd.to_datetime(f'{year}-06-09')).dt.days
    
    for i, mtn in enumerate(mountains):
        subset = season[season['mountain'] == mtn].sort_values('day_in_season')
        # shift x-axis by offset to separate years
        plt.plot(subset['day_in_season'] + offset, subset['avg_snowfall'], label=mtn if year == years[0] else "", color=plt.cm.tab10(i))
    
    # move offset for next year
    offset += year_width

# Adjust x-axis labels to show years
xticks = [year_width * i + year_width / 2 for i in range(len(years))]
plt.xticks(xticks, [str(y) for y in years])

plt.ylim(0, 25)  # set y-axis from 0 to 150

plt.xlabel('Season (June 9 → Sep 22) by Year')
plt.ylabel('Average Snowfall per Day (mm)')
plt.title('Average Snowfall Per Mountain Per Season')
plt.legend()
plt.tight_layout()
plt.show()



# print(df_join)
