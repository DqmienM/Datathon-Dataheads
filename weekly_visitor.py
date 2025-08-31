import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

df_vis_weekly = df_vis

# Start the ski season on 9th June
ski_season_start = pd.to_datetime(df_vis_weekly["Year"].astype(str) + "-06-09")
# Add seven days to each consequtive week
df_vis_weekly["Date"] = ski_season_start + pd.to_timedelta((df_vis_weekly["Week"]-1)*7, unit="D")

print(df_vis_weekly.to_string())

# Plot timeseries for each mountain
for mount in mountains:
    plt.figure(figsize = (10,6))
    plt.plot(df_vis_weekly["Date"], df_vis_weekly[mount], label = mount, color = "tab:blue")
    plt.title(f"Weekly Visitor Numbers at {mount} from 2014 - 2024")
    plt.xlabel("Date")
    plt.ylabel("Number of Visitors")
    plt.legend()
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.show()