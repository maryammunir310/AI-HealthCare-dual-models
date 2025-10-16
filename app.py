from fpdf import FPDF
# Add missing import for Streamlit
import streamlit as st

# --- Custom CSS Styling with Background Image ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Background image with medical theme */
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.92)), 
                          url('https://images.unsplash.com/photo-1559757175-0eb30cd8c063?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Title styling */
    .title-text {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: white !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card-like containers for each section */
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect>div>div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Sidebar radio buttons */
    .stRadio>div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(45deg, #4ECDC4, #44A08D) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        text-align: center !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 1rem 0 !important;
        box-shadow: 0 8px 32px rgba(78, 205, 196, 0.3) !important;
    }
    
    /* Section headers */
    h3 {
        color: #2c3e50 !important;
        border-bottom: 3px solid #4ECDC4 !important;
        padding-bottom: 0.5rem !important;
        font-weight: 700 !important;
    }
    
    /* List items styling */
    .stMarkdown li {
        background: rgba(78, 205, 196, 0.1) !important;
        margin: 0.3rem 0 !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #4ECDC4 !important;
    }
    
    /* Download button specific styling */
    .download-btn {
        background: linear-gradient(45deg, #27ae60, #2ecc71) !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(45deg, #27ae60, #2ecc71) !important;
        color: white !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
import importlib.util

st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='color: white; margin: 0;'>🏥 Personalized Medical Recommendation</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Medical Recommendation System", "Heart Disease Predictor"])

if page == "Medical Recommendation System":
    # ...existing code for medical recommendation system...
    import pandas as pd
    import pickle
    import numpy as np
    model = pickle.load(open("model/trained_model.pkl", "rb"))
    symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))
    description_df = pd.read_csv("datasets/description.csv")
    precautions_df = pd.read_csv("datasets/precautions_df.csv")
    diet_df = pd.read_csv("datasets/diets.csv")
    medications_df = pd.read_csv("datasets/medications.csv")
    workout_df = pd.read_csv("datasets/workout_df.csv")
    
    st.markdown('<div class="title-text">💊 WELCOME to Health Sight</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Select your symptoms and get a personalized medical recommendation.")
    user_symptoms = st.multiselect("Select Symptoms", symptoms_list)
    st.markdown('</div>', unsafe_allow_html=True)
    
    def get_column(info, col_name):
        if col_name in info.columns and pd.notna(info[col_name].values[0]):
            return [s.strip() for s in str(info[col_name].values[0]).split("|")]
        else:
            return ["No information available"]
    
    if st.button("🔍 Predict Disease"):
        if not user_symptoms:
            st.warning("Please select at least one symptom!")
        else:
            input_vector = np.zeros(len(symptoms_list))
            for symptom in user_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    input_vector[idx] = 1
            predicted_disease = model.predict([input_vector])[0]
            
            st.markdown(f'<div class="prediction-result">Predicted Disease: {predicted_disease}</div>', unsafe_allow_html=True)
            
            def normalize(s):
                return str(s).strip().lower()
            
            pred_norm = normalize(predicted_disease)
            info = description_df[description_df["Disease"].apply(normalize) == pred_norm]
            description = info["Description"].values[0] if (not info.empty and "Description" in info.columns) else "No description available."
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
            diet = ["No information available"]
            if "Disease" in diet_df.columns:
                diet_row = diet_df[diet_df["Disease"].apply(normalize) == pred_norm]
                if not diet_row.empty:
                    import ast
                    def parse_list_column(cell):
                        if pd.isna(cell) or cell == "":
                            return ["No information available"]
                        try:
                            return ast.literal_eval(cell)
                        except:
                            return [str(cell)]
                    diet = parse_list_column(diet_row["Diet"].values[0])
            medications = ["No information available"]
            if "Disease" in medications_df.columns:
                med_row = medications_df[medications_df["Disease"].apply(normalize) == pred_norm]
                if not med_row.empty:
                    import ast
                    def parse_list_column(cell):
                        if pd.isna(cell) or cell == "":
                            return ["No information available"]
                        try:
                            return ast.literal_eval(cell)
                        except:
                            return [str(cell)]
                    medications = parse_list_column(med_row["Medication"].values[0])
            workout = ["No information available"]
            workout_col = None
            for col in ["Disease", "disease"]:
                if col in workout_df.columns:
                    workout_col = col
                    break
            if workout_col:
                workout_row = workout_df[workout_df[workout_col].apply(normalize) == pred_norm]
                if not workout_row.empty:
                    w_col = "Workout" if "Workout" in workout_row.columns else ("workout" if "workout" in workout_row.columns else None)
                    if w_col:
                        import ast
                        def parse_list_column(cell):
                            if pd.isna(cell) or cell == "":
                                return ["No information available"]
                            try:
                                return ast.literal_eval(cell)
                            except:
                                return [str(cell)]
                        workout = parse_list_column(workout_row[w_col].values[0])
            
            # Wrap each section in a styled card
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Description")
            st.write(description)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 🛡️ Precautions")
            for i, item in enumerate(precautions):
                st.write(f"{i+1}. {item}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 🍎 Recommended Diet")
            for i, item in enumerate(diet):
                st.write(f"{i+1}. {item}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 💪 Recommended Workout/Exercise")
            for i, item in enumerate(workout):
                st.write(f"{i+1}. {item}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 💊 Suggested Medications")
            for i, item in enumerate(medications):
                st.write(f"{i+1}. {item}")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- PDF Download Button ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Personalized Medical Recommendation", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
            pdf.ln(5)
            pdf.multi_cell(0, 10, txt=f"Description: {description}")
            pdf.ln(5)
            pdf.cell(200, 10, txt="Precautions:", ln=True)
            for i, item in enumerate(precautions):
                pdf.cell(200, 10, txt=f"{i+1}. {item}", ln=True)
            pdf.ln(2)
            pdf.cell(200, 10, txt="Recommended Diet:", ln=True)
            for i, item in enumerate(diet):
                pdf.cell(200, 10, txt=f"{i+1}. {item}", ln=True)
            pdf.ln(2)
            pdf.cell(200, 10, txt="Recommended Workout/Exercise:", ln=True)
            for i, item in enumerate(workout):
                pdf.cell(200, 10, txt=f"{i+1}. {item}", ln=True)
            pdf.ln(2)
            pdf.cell(200, 10, txt="Suggested Medications:", ln=True)
            for i, item in enumerate(medications):
                pdf.cell(200, 10, txt=f"{i+1}. {item}", ln=True)
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            
            st.markdown('<div class="section-card" style="text-align: center;">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="medical_recommendation.pdf",
                mime="application/pdf"
            )
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Heart Disease Predictor":
    # Dynamically import and run the heart disease predictor app
    import sys
    import os
    import runpy
    heart_app_path = os.path.join(os.path.dirname(__file__), "heart_disease", "app.py")
    runpy.run_path(heart_app_path, run_name="__main__")