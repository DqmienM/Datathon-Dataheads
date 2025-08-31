import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_script import pred_visitors,df_join_weekly
from imblearn.over_sampling import SMOTE

# hello sussy bakas!

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler



WEEK_PERIOD = 15
df_join_weekly["week_sin"] = np.sin(2 * np.pi * df_join_weekly["Week"] / WEEK_PERIOD)
df_join_weekly["week_cos"] = np.cos(2 * np.pi * df_join_weekly["Week"] / WEEK_PERIOD)


features_num = ["max", "min", "rainfall", "week_sin", "week_cos"]
features_cat = ["mountain"]
target = "visitors"

X = df_join_weekly[features_num + features_cat]
y = df_join_weekly[target]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
    ],
    remainder="drop"
)

# Pipeline: preprocessing + model
model = Pipeline(steps=[
    ("prep", preprocess),
    ("est", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train/test split by year (so no leakage)
train = df_join_weekly["Year"] < 2024
X_train, X_test = X[train], X[~train]
y_train, y_test = y[train], y[~train]

# Fit
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R^2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


# new_data = pd.DataFrame({
#     "Year": [2025],
#     "Week": [2],
#     "mountain": ["Mt. Buller"],
#     "max": [2.5],
#     "min": [-3.0],
#     "rainfall": [5.0]
# })

# add cyclical features
df_join_weekly["week_sin"] = np.sin(2 * np.pi * df_join_weekly["Week"] / WEEK_PERIOD)
df_join_weekly["week_cos"] = np.cos(2 * np.pi * df_join_weekly["Week"] / WEEK_PERIOD)

# predict
pred = model.predict(df_join_weekly[features_num + features_cat])
print("Predicted visitors:", pred[0])