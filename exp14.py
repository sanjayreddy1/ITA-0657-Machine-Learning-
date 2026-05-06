import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target
print("Dataset Sample:\n")
print(df.head())
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Absolute Error:", round(mae, 3))
print("R2 Score:", round(r2, 3))
new_house = pd.DataFrame({
    "MedInc": [5.0],        
    "HouseAge": [20],
    "AveRooms": [6],
    "AveBedrms": [1],
    "Population": [1000],
    "AveOccup": [3],
    "Latitude": [34.0],
    "Longitude": [-118.0]
})
predicted_price = model.predict(new_house)
print("\nPredicted House Price:", round(predicted_price[0] * 100000, 2), "USD")
