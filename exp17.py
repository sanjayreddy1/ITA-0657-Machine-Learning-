import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    "battery_power": [842, 1021, 563, 615, 1821, 1859, 1954, 1445, 509, 769],
    "ram": [2549, 2631, 2603, 2769, 1411, 1067, 3230, 2948, 2828, 1447],
    "px_height": [20, 905, 1263, 1216, 1208, 1004, 381, 512, 386, 1137],
    "px_width": [756, 1988, 1716, 1786, 1212, 1654, 1018, 1149, 836, 1224],
    "mobile_wt": [188, 136, 145, 131, 141, 164, 131, 165, 113, 182],
    "talk_time": [19, 7, 9, 11, 15, 10, 18, 20, 5, 19],
    "dual_sim": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    "four_g": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    "price_range": [1, 2, 2, 2, 3, 1, 3, 3, 1, 2]  # target (0=low, 3=high)
}
df = pd.DataFrame(data)
print("Dataset Sample:\n")
print(df.head())
X = df.drop("price_range", axis=1)
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_mobile = pd.DataFrame({
    "battery_power": [1500],
    "ram": [3000],
    "px_height": [1000],
    "px_width": [1500],
    "mobile_wt": [150],
    "talk_time": [15],
    "dual_sim": [1],
    "four_g": [1]
})
prediction = model.predict(new_mobile)
print("\nPredicted Price Range (0=Low, 3=High):", prediction[0])
