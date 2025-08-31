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

df_vis_weekly = df_vis.copy() #should work lmao

# print(df_vis_weekly)


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

df_cli_weekly = df_cli.copy()

ANCHOR_MONTH = 6
ANCHOR_DAY = 9

# Ensure datetime
df_cli_weekly["date"] = pd.to_datetime(df_cli_weekly["date"], errors="coerce")

# Build the "June 9 of the same calendar year" for each row
anniv_this_year = pd.to_datetime({
    "year": df_cli_weekly["date"].dt.year,
    "month": ANCHOR_MONTH,
    "day": ANCHOR_DAY
})

# Season year: on/after June 9 → same year; before → previous year
df_cli_weekly["season_year"] = np.where(df_cli_weekly["date"] >= anniv_this_year,
                             df_cli_weekly["date"].dt.year,
                             df_cli_weekly["date"].dt.year - 1)

# Week-1 start for that season (June 9 of season_year)
week1_start = pd.to_datetime({
    "year": df_cli_weekly["season_year"],
    "month": ANCHOR_MONTH,
    "day": ANCHOR_DAY
})

# Relative week number (June 9..15 = 1; June 16..22 = 2; etc.)
df_cli_weekly["rel_week"] = ((df_cli_weekly["date"] - week1_start).dt.days // 7) + 1

CUTOFF = pd.Timestamp("2014-06-09")  # your cutoff
df_cli_weekly = df_cli_weekly[df_cli_weekly["date"] >= CUTOFF]

# print(df_cli_weekly.head(20))

# Example value columns
agg_map = {"max": "mean", "min": "mean", "rainfall": "mean"}  # change as needed
by_cols = ["season_year", "rel_week", "mountain"]

# If you also need per-station weekly stats, include "station" in by_cols
weekly = (
    df_cli_weekly.groupby(by_cols, as_index=False)
      .agg(agg_map)
)

# print(weekly.head(20))

# Add the actual start date of each week (handy for filtering or joining)
weekly["week_start"] = (
    pd.to_datetime({
        "year": weekly["season_year"],
        "month": ANCHOR_MONTH,
        "day": ANCHOR_DAY
    })
    + pd.to_timedelta((weekly["rel_week"] - 1) * 7, unit="D")
)

# print(weekly.head(20))

# Optional: order columns
df_cli_weekly = weekly[["season_year", "rel_week", "week_start", "mountain"] + list(agg_map.keys())]

df_cli_weekly = df_cli_weekly.rename(columns={
    "season_year": "Year",
    "rel_week": "Week"
})

# print("\n\ndf cli weekly\n\n",df_cli_weekly.head(20))

#-----------------------

# print(df_cli)

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)
# df_vis.plot()
# print("\n\n")

# print(df_vis.head(120))
# print(df_cli.head(120))

#Join is uniquely identified through station_no and date i thinkmien
pred_visitors = pd.merge(df_vis, df_cli, on=["date"], how="inner")

df_join_weekly = pd.merge(df_vis_weekly, df_cli_weekly, on=["Year","Week"], how="inner")

#turn the corresponding mountain into a visitors
pred_visitors["visitors"] = pred_visitors.apply(lambda row: row[row["mountain"]]/7, axis=1)

pred_visitors = pred_visitors.drop(mountains, axis=1)
pred_visitors = pred_visitors.drop("Week", axis=1)

# print(df_join_weekly.columns.tolist())
#turn the corresponding mountain into a visitors
df_join_weekly["visitors"] = df_join_weekly.apply(lambda row: row[row["mountain"]], axis=1)

df_join_weekly = df_join_weekly.drop(mountains, axis=1)
df_join_weekly = df_join_weekly.drop("week_start", axis=1)
# df_join_weekly = df_join_weekly.drop("date", axis=1)
# print(df_join.columns.tolist())

# df_climate = df_climate.rename(columns={
#     "station_no": "mountain",
# })

pred_visitors["avg_visitors_over_next_week"] = pred_visitors["visitors"]

pred_visitors = pred_visitors.dropna()
df_join_weekly = df_join_weekly.dropna()


# take the date and change average visitors over the next week
# I want to take current date and current mountain and then take the next 6 days and sum them and divide by the number of days available
# (may be a day with no data points so not always divide by 7)

# for index, row in df_join.iterrows():
#     print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}")



# print(df_join[["date","mountain"]])
# print(df_join)
# print(df_join_weekly.head(30))


pred_visitors.to_csv("visitor_climate_joined_daily.csv",index=False)
df_join_weekly.to_csv("visitor_climate_joined_weekly.csv",index=False)