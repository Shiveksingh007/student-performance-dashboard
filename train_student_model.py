import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

os.chdir(r'd:\pro\Student_Preformance')

# Load and preprocess data
df = pd.read_csv("data/student_performance_200.csv")
df['result'] = df['result'].str.strip().str.lower().map({'fail': 0, 'pass': 1})
df = df.dropna()

# Train model
X = df.drop('result', axis=1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Display results
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

# Save model files
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/student_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and Scaler saved!")