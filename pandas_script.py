import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

#create a datetime for the climate table
df_cli["date"] = pd.to_datetime(df_cli[["Year", "Month", "Day"]], errors="coerce")


week1_start = pd.to_datetime( df_vis["Year"].astype(str) + "-01-01") 
df_vis["date"] = pd.to_datetime(df_vis[["Year"]], errors="coerce" )

#print(df_vis.to_string())
#print(df_vis.info())
#print(df_vis.describe())
#print(df_vis.shape)
df_vis.plot()
print("\n\n")

print(df_cli.to_string())




