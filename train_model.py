if __name__ == "__main__":
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Load training data
    train_path = os.path.join("datasets", "Training.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        X = df.drop("prognosis", axis=1)
        y = df["prognosis"]
        symptoms_list = list(X.columns)

        # Split for quick validation (optional)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model and symptoms list
        with open("model/trained_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("model/symptoms_list.pkl", "wb") as f:
            pickle.dump(symptoms_list, f)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model retrained and saved. Test accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred))
    else:
        print("Training.csv not found for retraining.")
# --- Model Evaluation (run this part to check model accuracy) ---
import os
if __name__ == "__main__":
    import pickle
    import numpy as np
    import pandas as pd
    # Load model and symptom list for evaluation
    model = pickle.load(open("model/trained_model.pkl", "rb"))
    symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))
    # Evaluate model accuracy on Training.csv
    train_path = os.path.join("datasets", "Training.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        X = df.drop("prognosis", axis=1)
        y = df["prognosis"]
        # If model expects label encoding, ensure y is encoded as in training
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        # Predict
        y_pred = model.predict(X)
        # If model output is encoded, decode for comparison
        if hasattr(model, "classes_"):
            y_pred_decoded = [model.classes_[i] if isinstance(i, (int, np.integer)) else i for i in y_pred]
        else:
            y_pred_decoded = y_pred
        # Accuracy
        def normalize(s):
            return str(s).strip().lower()
        acc = np.mean([normalize(a) == normalize(b) for a, b in zip(y, y_pred_decoded)])
        print(f"Model accuracy on training data: {acc*100:.2f}%")
    else:
        print("Training.csv not found for evaluation.")
# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import ast  # for converting stringified lists to actual lists

# --- Load model and symptom list ---
model = pickle.load(open("model/trained_model.pkl", "rb"))
symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))

# --- Load datasets ---
description_df = pd.read_csv("datasets/description.csv")
precautions_df = pd.read_csv("datasets/precautions_df.csv")
diet_df = pd.read_csv("datasets/diets.csv")
medications_df = pd.read_csv("datasets/medications.csv")
workout_df = pd.read_csv("datasets/workout_df.csv")

# Function to safely convert stringified list to Python list
def parse_list_column(cell):
    if pd.isna(cell) or cell == "":
        return ["No information available"]
    try:
        return ast.literal_eval(cell)
    except:
        return [str(cell)]

# --- Streamlit UI ---
st.title("💊 Personalized Medical Recommendation System")
st.write("Select your symptoms and get a personalized medical recommendation.")

user_symptoms = st.multiselect("Select Symptoms", symptoms_list)

if st.button("Predict Disease"):
    if not user_symptoms:
        st.warning("Please select at least one symptom!")
    else:
        # Create input vector
        input_vector = np.zeros(len(symptoms_list))
        for symptom in user_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1

        # Predict disease
        predicted_disease = model.predict([input_vector])[0]
        st.subheader(f"Predicted Disease: {predicted_disease}")


        # Normalize disease names for robust matching
        def normalize(s):
            return str(s).strip().lower()

        pred_norm = normalize(predicted_disease)

        # Description
        desc_row = description_df[description_df["Disease"].apply(normalize) == pred_norm]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

        # Precautions
        precautions = ["No information available"]
        if "Disease" in precautions_df.columns:
            prec_row = precautions_df[precautions_df["Disease"].apply(normalize) == pred_norm]
            if not prec_row.empty:
                temp = []
                for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
                    if col in prec_row:
                        val = prec_row[col].values[0]
                        if pd.notna(val) and str(val).strip() != "":
                            temp.append(val)
                if temp:
                    precautions = temp

        # Diet
        diet = ["No information available"]
        if "Disease" in diet_df.columns:
            diet_row = diet_df[diet_df["Disease"].apply(normalize) == pred_norm]
            if not diet_row.empty:
                diet = parse_list_column(diet_row["Diet"].values[0])

        # Medications
        medications = ["No information available"]
        if "Disease" in medications_df.columns:
            med_row = medications_df[medications_df["Disease"].apply(normalize) == pred_norm]
            if not med_row.empty:
                medications = parse_list_column(med_row["Medication"].values[0])

        # Workout/Exercise
        workout = ["No information available"]
        # Try both 'Disease' and 'disease' columns for robustness
        workout_col = None
        for col in ["Disease", "disease"]:
            if col in workout_df.columns:
                workout_col = col
                break
        if workout_col:
            workout_row = workout_df[workout_df[workout_col].apply(normalize) == pred_norm]
            if not workout_row.empty:
                # Try both 'Workout' and 'workout' columns
                w_col = "Workout" if "Workout" in workout_row.columns else ("workout" if "workout" in workout_row.columns else None)
                if w_col:
                    workout = parse_list_column(workout_row[w_col].values[0])

        # --- Display ---
        st.markdown("### Description")
        st.write(description)

        st.markdown("### Precautions")
        for i, item in enumerate(precautions):
            st.write(f"{i+1}. {item}")

        st.markdown("### Recommended Diet")
        for i, item in enumerate(diet):
            st.write(f"{i+1}. {item}")

        st.markdown("### Suggested Medications")
        for i, item in enumerate(medications):
            st.write(f"{i+1}. {item}")

        st.markdown("### Recommended Workout/Exercise")
        for i, item in enumerate(workout):
            st.write(f"{i+1}. {item}")

# --- REAL MODEL METRICS (from test split) ---
print("\n" + "="*50)
print("REAL TEST PERFORMANCE (on unseen data)")
print("="*50)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Use the same test split from above
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"✅ Test Accuracy:  {accuracy*100:.2f}%")
print(f"✅ Test Precision: {precision*100:.2f}%")
print(f"✅ Test Recall:    {recall*100:.2f}%")
print(f"✅ Test F1-Score:  {f1*100:.2f}%")
print(f"✅ Classes:        {len(model.classes_)} diseases")
print(f"✅ Features:       {len(symptoms_list)} symptoms")