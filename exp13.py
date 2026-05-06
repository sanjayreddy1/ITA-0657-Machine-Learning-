import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    "Car_Name": ["ritz", "sx4", "ciaz", "wagonr", "swift", "vitara", "baleno", "verna", "city", "brio"],
    "Year": [2014, 2013, 2017, 2011, 2014, 2018, 2016, 2015, 2017, 2013],
    "Selling_Price": [3.35, 4.75, 7.25, 2.85, 4.60, 9.25, 6.75, 5.25, 8.50, 3.65],
    "Present_Price": [5.59, 9.54, 9.85, 4.15, 6.87, 12.50, 8.75, 7.80, 11.20, 5.40],
    "Kms_Driven": [27000, 43000, 6900, 52000, 42450, 15000, 22000, 35000, 12000, 48000],
    "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel", "Diesel", "Petrol", "Diesel", "Petrol", "Petrol"],
    "Seller_Type": ["Dealer", "Dealer", "Dealer", "Individual", "Dealer", "Dealer", "Individual", "Dealer", "Dealer", "Individual"],
    "Transmission": ["Manual", "Manual", "Manual", "Manual", "Manual", "Automatic", "Manual", "Automatic", "Manual", "Manual"],
    "Owner": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
df = pd.DataFrame(data)
current_year = 2024
df["Car_Age"] = current_year - df["Year"]
df.drop(["Car_Name", "Year"], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
new_data = pd.DataFrame({
    "Present_Price": [7.0],
    "Kms_Driven": [30000],
    "Owner": [0],
    "Car_Age": [5],
    "Fuel_Type_Petrol": [1],
    "Seller_Type_Individual": [0],
    "Transmission_Manual": [1]
})
new_data = new_data.reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(new_data)

print("\nPredicted Price:", round(predicted_price[0], 2), "lakhs")
