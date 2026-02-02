import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



columns = ["IQ", "CGPA", "Tenth", "Twelfth", "Comm_Skill", 
           "Tech_Skills", "Comm", "Hackathons", "Certifications", "Backlogs","Interview_Performance"]

np.random.seed(42)
n_samples = 2000
data = {
    'IQ': np.random.randint(50, 140, n_samples),
    'CGPA': np.random.uniform(5.0, 10.0, n_samples),
    'Tenth': np.random.randint(50, 95, n_samples),
    'Twelfth': np.random.randint(50, 95, n_samples),
    'Comm_Skill': np.random.randint(1, 10, n_samples),
    'Tech_Skills': np.random.randint(1, 10, n_samples),
    'Comm': np.random.randint(1, 10, n_samples),
    'Hackathons': np.random.randint(0, 5, n_samples),
    'Certifications': np.random.randint(0, 5, n_samples),
    'Backlogs': np.random.randint(0, 4, n_samples),
    'Interview_Performance': np.round(
        (np.random.randint(1, 10, n_samples) +
         np.random.randint(1, 10, n_samples)) / 2, 1)
}
df = pd.DataFrame(data)


def placement_logic(row):
    score = 0

    
    if row['Tenth'] >= 60 and row['Twelfth'] >= 60:
        score += 25
    else:
        score -= 40   

    score += (row['CGPA'] * 6)

  
    score += (row['Tech_Skills'] * 8)          
    score += (row['Comm_Skill'] * 5)           
    score += (row['Interview_Performance'] * 15)

    score += (row['Hackathons'] * 4)
    score += (row['Certifications'] * 3)

   
    if row['IQ'] < 50:
        score -= 20

    score -= (row['Backlogs'] * 20)

   
    return 1 if score >= 180 else 0

df['Placed'] = df.apply(placement_logic, axis=1)

X = df[columns]
y = df['Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print(f"✅ Model Trained! Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\n--- 🔍 SCENARIO CHECK ---")


safe_iq_student = [[60, 8.5, 75, 75, 9, 8, 9, 3, 2, 0,8.5]]
pred = model.predict(safe_iq_student)[0]
print(f"Student with IQ 70 (Safe Zone): {'🟢 Placed' if pred==1 else '🔴 Not Placed'}")


critical_iq_student = [[40, 8.5, 75, 75, 9, 8, 9, 3, 2, 0,8.5]]
pred2 = model.predict(critical_iq_student)[0]
print(f"Student with IQ 55 (Critical Zone): {'🟢 Placed' if pred2==1 else '🔴 Not Placed'}")

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n💾 Saved to model.pkl")