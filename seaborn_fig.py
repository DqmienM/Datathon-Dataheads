import matplotlib.pyplot as plt
import pandas as pd


# import data
df_vis = pd.read_csv('visitation_data.csv')
mountains = ["Mt. Baw Baw","Mt. Stirling","Mt. Hotham","Falls Creek","Mt. Buller","Selwyn","Thredbo","Perisher","Charlotte Pass"]

# filtering out null columns
df_vis = df_vis.dropna()

# converting values to better forms
df_vis["Year"] = df_vis["Year"].astype(int)
df_vis["Year"] = df_vis["Year"].astype(str)

df_vis["Week"] = df_vis["Week"].astype(int)


# format data for each mountain
mountain_data = []
for mountain in mountains:
    data = {}
    # get the years
    data["Year"] = sorted(list(set(df_vis.to_dict()["Year"].values())))

    # add each week to the data dict
    for i in range(1,16):
        week = df_vis[df_vis.Week == i]
        data[f"Week {i}"] = list(week.to_dict()[mountain].values())

    # add data to list
    mountain_data.append(data)


# plot data for each mountain
for i, mountain in enumerate(mountains):
    df_new = pd.DataFrame(mountain_data[i])
    df_new.plot(x='Year', kind='bar', stacked=False, title=mountain, subplots=True)


plt.show()