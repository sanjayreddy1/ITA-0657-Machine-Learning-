import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------
data = {
    'income': [15000, 30000, 45000, 60000, 80000, 120000, 20000, 50000],
    'age': [25, 35, 45, 50, 30, 40, 23, 37],
    'loan_amount': [5000, 10000, 15000, 20000, 12000, 25000, 3000, 18000],
    'credit_score': ['Low', 'Medium', 'High', 'High', 'Medium', 'High', 'Low', 'Medium']
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Encode Labels
# -----------------------------
le = LabelEncoder()
df['credit_score'] = le.fit_transform(df['credit_score'])  
# Low=0, Medium=1, High=2

# -----------------------------
# Step 3: Split Data
# -----------------------------
X = df[['income', 'age', 'loan_amount']]
y = df['credit_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Step 7: Test with New Data
# -----------------------------
new_data = [[55000, 32, 15000]]
prediction = model.predict(new_data)

print("\nNew Customer Prediction:",
      le.inverse_transform(prediction))