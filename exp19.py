import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
data = {
    "Gender": ["Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Married": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate", "Graduate"],
    "Self_Employed": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
    "ApplicantIncome": [5000, 3000, 6000, 4500, 3500, 8000, 4000, 7000, 3200, 6500],
    "LoanAmount": [200, 100, 250, 150, 120, 300, 180, 280, 110, 240],
    "Credit_History": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    "Loan_Status": ["Y", "N", "Y", "Y", "N", "Y", "N", "Y", "N", "Y"]
}
df = pd.DataFrame(data)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))  
