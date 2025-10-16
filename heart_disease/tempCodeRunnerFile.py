import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
heart_data = pd.read_csv(r"C:\Users\Tech Planet 3rd 84a\Downloads\heart_disease_data.csv")

# Features & target
X = heart_data.drop(columns='target')
Y = heart_data['target']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save model as pickle
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as heart_model.pkl")
