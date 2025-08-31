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


# Features for temperature prediction
X_temp = df_join_weekly[['Year', 'Week', 'mountain', 'rainfall']]  # Use rainfall if relevant
categorical_features = ['mountain']
numeric_features = ['Year', 'Week', 'rainfall']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Predict max temperature
y_max = df_join_weekly['max']
model_max = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_max.fit(X_temp, y_max)

# Predict min temperature
y_min = df_join_weekly['min']
model_min = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_min.fit(X_temp, y_min)

# Prepare 2026 data
weeks = range(1, 16)
mountains = df_join_weekly['mountain'].unique()
pred_data = []

for mountain in mountains:
    for week in weeks:
        # Rainfall could be set to past averages or np.nan if unknown
        pred_data.append({'Year': 2026, 'Week': week, 'mountain': mountain, 'rainfall': np.nan})

pred_df = pd.DataFrame(pred_data)

# Fill rainfall with past average if unknown
avg_rainfall = df_join_weekly.groupby('mountain')['rainfall'].mean()
for mountain in mountains:
    pred_df.loc[pred_df['mountain']==mountain, 'rainfall'] = avg_rainfall.loc[mountain]

# Predict max and min temperatures
pred_df['max'] = model_max.predict(pred_df)
pred_df['min'] = model_min.predict(pred_df)

df_2026 = pred_df

print(df_2026.head(50))


# Plot

# Assuming your 2026 data is in df_2026
weeks = sorted(df_2026['Week'].unique())
mountains = df_2026['mountain'].unique()

plt.figure(figsize=(14,6))

for i, week in enumerate(weeks, start=1):
    week_subset = df_2026[df_2026['Week'] == week]
    if not week_subset.empty:
        # get min and max across all mountains
        min_temp = week_subset['min'].min()
        max_temp = week_subset['max'].max()
        plt.vlines(x=i, ymin=min_temp, ymax=max_temp, color='blue', linewidth=5)

plt.xticks(range(1, len(weeks)+1), [f"W{w}" for w in weeks])
plt.xlabel("Week (2026)")
plt.ylabel("Temperature (°C)")
plt.title("Weekly Min-Max Temperature Range Predictions for 2026")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()