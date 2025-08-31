import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_script import pred_visitors,df_join_weekly
import seaborn as sns

corr_matrix = df_join_weekly.select_dtypes(include=np.number).corr()
print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm", center=0,fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()