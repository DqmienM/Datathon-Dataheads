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

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)
# df_vis.plot()
print("\n\n")

# print(df_vis.head(120))
print(df_cli.head(120))

#Join is uniquely identified through station_no and date i thinkmien
df_join = pd.merge(df_vis, df_cli, on="date", how="inner")

print(df_join.columns.tolist())

print(df_join[["Year_x", "date","station_no"]])


