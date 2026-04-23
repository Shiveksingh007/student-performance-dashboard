import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting training process...")

# 1. Load the data 
df = pd.read_csv("student_performance_200.csv")

# 2. Clean the target column
df['result'] = df['result'].str.strip().str.lower().map({'fail': 0, 'pass': 1})
df = df.dropna()

# 3. Split data
X = df.drop('result', axis=1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale and Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. SAVE THE FILES
joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Success! student_model.pkl and scaler.pkl have been created.")