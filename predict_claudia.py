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
svr_model = SVR(
    kernel="poly",
    degree=10,
    C=100,
    coef0=9
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", svr_model)
])

# Train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
pipeline.fit(X_train, y_train)

# Predict
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

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
plt.title("SVR: Predicted vs Actual Visitors (Test Set)")
plt.show()