import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_script import pred_visitors,df_join_weekly
from imblearn.over_sampling import SMOTE


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler



# features
X = df_join_weekly[["max", "min", "rainfall","Year","Week"]]
# target  
y = df_join_weekly["visitors"]    



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Oversampling
smote = SMOTE(random_state = 10)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#choose the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)



# evalutate
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("R^2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


#predict

print(model.predict(df_join_weekly))