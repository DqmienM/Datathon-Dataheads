import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from helpers import show_scatterplot
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

#Filtering out weeks with 0 visitors
df_vis = df_vis[(df_vis[mountains] != 0).any(axis=1)]

#filtering out null columns
df_vis = df_vis.dropna()

#converting values to int
df_vis["Year"] = df_vis["Year"].astype(int)
df_vis["Week"] = df_vis["Week"].astype(int)
for mount in mountains:
    df_vis[mount] = df_vis[mount].astype(int)

#create a datetime for the climate table
df_cli["date"] = pd.to_datetime(df_cli[["Year", "Month", "Day"]], errors="coerce")


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
df_climate = pd.concat([df_cli, df_avg], ignore_index=True)

# print(df_cli)

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)
# df_vis.plot()
print("\n\n")

# print(df_vis.head(120))
# print(df_cli.head(120))

#Join is uniquely identified through station_no and date i thinkmien
df_join = pd.merge(df_vis, df_cli, on=["date"], how="inner")

#turn the corresponding mountain into a visitors
df_join["visitors"] = df_join.apply(lambda row: row[row["mountain"]]/7, axis=1)

df_join = df_join.drop(mountains, axis=1)
df_join = df_join.drop("Week", axis=1)

print(df_join.columns.tolist())

# df_climate = df_climate.rename(columns={
#     "station_no": "mountain",
# })

# print(df_join[["date","mountain"]])
print(df_join)


