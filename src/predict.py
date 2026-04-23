import joblib
import pandas as pd

model = joblib.load("model/student_model.pkl")

features = ['study_hours', 'attendance', 'previous_score', 'assignments_completed']
data = []

for f in features:
    val = float(input(f"Enter {f}: "))
    data.append(val)

df = pd.DataFrame([data], columns=features)

pred = model.predict(df)[0]

print("PASS ✅" if pred == 1 else "FAIL ❌")