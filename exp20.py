import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    "Month": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Marketing_Spend": [2000, 2200, 2500, 2700, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4500],
    "Seasonality_Index": [1.0, 1.1, 1.2, 1.0, 1.3, 1.4, 1.5, 1.6, 1.2, 1.3, 1.1, 1.5],
    "Sales": [15000, 16000, 18000, 17000, 20000, 22000, 24000, 26000, 23000, 25000, 24000, 27000]
}
df = pd.DataFrame(data)
print("Dataset:\n")
print(df)
X = df.drop("Sales", axis=1)
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))
future_data = pd.DataFrame({
    "Month": [13, 14, 15],
    "Marketing_Spend": [4600, 4800, 5000],
    "Seasonality_Index": [1.3, 1.4, 1.6]
})
future_sales = model.predict(future_data)
print("\nPredicted Future Sales:")
for i, sale in enumerate(future_sales):
    print(f"Month {13+i}: {round(sale, 2)}")
