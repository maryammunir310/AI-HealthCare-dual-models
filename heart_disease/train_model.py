# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Load dataset ---
heart_data = pd.read_csv(r"C:\Users\Tech Planet 3rd 84a\Downloads\heart_disease_data.csv")

# --- 2. Split features and target ---
X = heart_data.drop(columns='target')
Y = heart_data['target']

# --- 3. Feature scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Train/test split ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# --- 5. Train Logistic Regression model with more iterations ---
model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_train, Y_train)

# --- 6. Save the trained model and scaler ---
with open("heart_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("✅ Model trained and saved as heart_model.pkl")

# --- 7. MODEL PERFORMANCE EVALUATION ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Predict on test data
Y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("\n" + "="*50)
print("HEART DISEASE MODEL PERFORMANCE")
print("="*50)
print(f"✅ Test Accuracy:  {accuracy*100:.2f}%")
print(f"✅ Test Precision: {precision*100:.2f}%")
print(f"✅ Test Recall:    {recall*100:.2f}%")
print(f"✅ Test F1-Score:  {f1*100:.2f}%")
print(f"✅ Dataset Size:   {len(heart_data)} patients")
print(f"✅ Features:       {X.shape[1]} health parameters")
print(f"✅ Train Samples:  {X_train.shape[0]}")
print(f"✅ Test Samples:   {X_test.shape[0]}")

# Detailed report
print("\n📊 Classification Report:")
print(classification_report(Y_test, Y_pred, target_names=['No Heart Disease', 'Heart Disease']))

# Confusion matrix
print("🎯 Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))