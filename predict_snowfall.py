import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_script import pred_visitors,df_join_weekly
from imblearn.over_sampling import SMOTE


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Features and target
X = df_join_weekly[['Year', 'Week', 'mountain', 'max', 'min']]  # exclude rainfall
y = df_join_weekly['rainfall']  # target is now rainfall

# Encode categorical 'mountain'
categorical_features = ['mountain']
numeric_features = ['Year', 'Week', 'max', 'min']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Random forest pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train on all past data
model.fit(X, y)

# Predict for 2026
weeks = range(1, 16)  # all weeks
mountains = df_join_weekly['mountain'].unique()
pred_data = []

for mountain in mountains:
    for week in weeks:
        pred_data.append({'Year': 2026, 'Week': week, 'mountain': mountain,
                          'max': np.nan, 'min': np.nan})

pred_df = pd.DataFrame(pred_data)

# Fill max/min with past averages
avg_weather = df_join_weekly.groupby('mountain')[['max','min']].mean()
for mountain in mountains:
    for col in ['max','min']:
        pred_df.loc[pred_df['mountain']==mountain, col] = avg_weather.loc[mountain, col]

# Predict rainfall
pred_df['rainfall'] = model.predict(pred_df)

# print(pred_df.head(50))

df_2026 = pred_df

# Plot
plt.figure(figsize=(12,6))

mountains = df_2026['mountain'].unique()
for mountain in mountains:
    subset = df_2026[df_2026['mountain'] == mountain]
    plt.plot(subset['Week'], subset['rainfall'], marker='o', label=mountain)

plt.xlabel('Week')
plt.ylabel('Snowfall (mm)')
plt.title('Predicted Snowfall per Week for 2026')
plt.xticks(range(1, max(df_2026['Week'])+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


