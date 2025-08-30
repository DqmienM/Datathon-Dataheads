import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from helpers import show_scatterplot
import seaborn as sns

df_cli = pd.read_csv('climate_data.csv')
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

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)

df_melted = df_vis.melt(id_vars=["Year", "Week"], value_vars=mountains, var_name="Mountain", value_name="Visitors")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_melted, x="Week", y="Visitors", hue="Mountain")
plt.title("Weekly Visitors by Mountain")
plt.xlabel("Week Number")
plt.ylabel("Visitors")
plt.grid(True)
plt.tight_layout()
plt.show()

