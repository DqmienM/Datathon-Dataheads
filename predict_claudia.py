import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_script import pred_visitors, df_join_weekly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from datetime import datetime, timedelta

# features
X = df_join_weekly[["max", "min", "rainfall","Week", "mountain"]]
# target  
y = df_join_weekly["visitors"]    


quantitative_features = ["max", "min", "rainfall", "Week"]
qualitative_features = ["mountain"]


preprocessor = ColumnTransformer(
    transformers=[
        ("quantitative", StandardScaler(), quantitative_features),
        ("qualitative", OneHotEncoder(drop="first", sparse_output=False), qualitative_features)
    ]
)

# Choose model
svr_model = SVR(kernel="rbf", C=100, gamma="scale")

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(regressor=svr_model, transformer=StandardScaler()))
])

# Train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
pipeline.fit(X_train, y_train)

# Predict existing data
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Test model reliability
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train MSE: {mse_train:.2f}, R^2: {r2_train:.3f}")
print(f"Test  MSE: {mse_test:.2f}, R^2: {r2_test:.3f}")

# Plot
plt.figure(figsize=(10,8))
plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Visitors")
plt.ylabel("Predicted Visitors")
plt.title("Predicted vs Actual Visitors Numbers for Test Data")
plt.show()


# PREDICT 2026 
def generate_season_weeks(year, start_month=6, start_day=9, end_month=9, end_day=15):
    start = datetime(year, start_month, start_day)
    end = datetime(year, end_month, end_day)
    weeks = []
    week = 1
    while start <= end:
        weeks.append({"Year": year, "Week": week, "week_start": start})
        start += timedelta(days=7)
        week += 1
    return pd.DataFrame(weeks)
df_2026_weeks = generate_season_weeks(2026)

climatepredict = (
    df_join_weekly.groupby(["mountain", "Week"])[["max","min","rainfall"]]
    .mean()
    .reset_index()
)
df_2026 = df_2026_weeks.merge(climatepredict, on="Week")

X_2026 = df_2026[["max", "min", "rainfall", "Week", "mountain"]]
df_2026["predicted_visitors"] = pipeline.predict(X_2026)
df_2026.to_csv("predicted_visitors_2026.csv", index=False)


# Plot
df_2026["week_label"] = "Week " + df_2026["Week"].astype(str)
plt.figure(figsize=(12,10))
for mount in df_2026["mountain"].unique():
    subset = df_2026[df_2026["mountain"] == mount]
    plt.plot(subset["week_label"], subset["predicted_visitors"], marker="o", label=mount)
plt.legend()
plt.title("Predicted Visitors per Week for 2026")
plt.xlabel("Weeks")
plt.ylabel("Visitors")
plt.show()