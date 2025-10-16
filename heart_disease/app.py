# app.py
import streamlit as st
import pandas as pd
import pickle
import altair as alt
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from datetime import datetime

# --- Load trained model and scaler ---
import os
try:
    model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- Streamlit page config ---
st.set_page_config(
    page_title="CardioGuard ❤️",
    page_icon=":heart:",
    layout="centered"
)

# --- CSS Styling with Background Image ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.8rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.6) !important;
    }
    
    .main-title {
        font-size: 4rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(45deg, #FF0066, #FF8C00, #FF0066);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .stSelectbox>div>div, .stSlider>div>div, .stNumberInput>div>div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .prediction-healthy {
        background: linear-gradient(45deg, #00b09b, #96c93d) !important;
        color: white !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        text-align: center !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(0, 176, 155, 0.4) !important;
    }
    
    .prediction-risk {
        background: linear-gradient(45deg, #ff416c, #ff4b2b) !important;
        color: white !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        text-align: center !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.4) !important;
    }
    
    .tips-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2) !important;
    }
    
    .download-btn {
        background: linear-gradient(45deg, #27ae60, #2ecc71) !important;
    }
    
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    .welcome-header {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# --- Neon Title ---
st.markdown("""
<div class="main-title">
CardioGuard <span style='font-size: 4rem;'>❤️</span>
</div>
""", unsafe_allow_html=True)

# --- Name input and welcome ---
name = st.text_input("Enter your name:", "")
if name:
    st.markdown(f"<div class='welcome-header'>💖 Hello <span style='color:#ffeb3b'>{name}</span>! Let's check your heart health ❤️</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.3rem; color:#666;'>Wishing you a healthy heart journey! 🫀✨</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='welcome-header'>Welcome! Let's check your heart health ❤️</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.3rem; color:#666;'>Please enter your name for a personalized experience.</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Options for categorical inputs ---
sex_options = {0: "Female", 1: "Male"}
cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
fbs_options = {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"}
restecg_options = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
exang_options = {0: "No", 1: "Yes"}
slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

# --- User health input ---
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 🩺 Enter Your Health Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 45)
        sex = st.selectbox("Sex", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
        cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()), format_func=lambda x: fbs_options[x])
        restecg = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
    with col2:
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()), format_func=lambda x: exang_options[x])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
        ca = st.slider("Number of Major Vessels (ca)", 0, 4, 0)
        thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# Store prediction results for PDF
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# --- Analyze button ---
if st.button("🔍 Analyze Heart Health", use_container_width=True):
    try:
        # Prepare input data
        input_df = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Prediction
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        prob_percent = [round(p * 100, 2) for p in prob]

        # Store data for PDF
        st.session_state.prediction_data = {
            'name': name if name else "User",
            'prediction': pred,
            'risk_percent': prob_percent[1],
            'healthy_percent': prob_percent[0],
            'input_data': input_df.iloc[0].to_dict(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # --- Prediction result card ---
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        user_name = name if name else "User"
        if pred == 0:
            st.markdown(f"<div class='prediction-healthy'>💚 {user_name}, your heart appears healthy!</div>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"<div class='prediction-risk'>⚠️ {user_name}, you may have heart disease. Please consult a doctor!</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Gauge Chart for Risk ---
        risk_percent = prob_percent[1]
        healthy_percent = prob_percent[0]
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
                'bar': {'color': "#FF0000" if risk_percent > 50 else "#00FF00"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 30], 'color': '#e6ffe6'},
                    {'range': [30, 70], 'color': '#fffbe6'},
                    {'range': [70, 100], 'color': '#ffe6e6'}
                ],
                'threshold': {
                    'line': {'color': "#d72660", 'width': 6},
                    'thickness': 0.8,
                    'value': risk_percent
                }
            },
            number={'suffix': "%"}
        ))
        gauge_fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # --- Risk Label ---
        if risk_percent < 30:
            st.markdown("<div style='text-align:center;font-size:28px;color:#00b300;font-weight:700;'>Low Risk 💚</div>", unsafe_allow_html=True)
        elif risk_percent < 70:
            st.markdown("<div style='text-align:center;font-size:28px;color:#ffae42;font-weight:700;'>Moderate Risk 🟡</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center;font-size:28px;color:#d72660;font-weight:700;'>High Risk ❤️‍🔥</div>", unsafe_allow_html=True)

        # --- Heart Care Tips Card ---
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown("### 💡 Heart Care Tips")
        if pred == 0:
            tips_html = """
            <div class='tips-card'>
                <b style='font-size:26px;'>💚 Tips for a Healthy Heart</b><br><br>
                <span style='font-size:20px;'>🥗 Eat Healthy &nbsp; | &nbsp; 🏃‍♂️ Exercise &nbsp; | &nbsp; 🚭 No Smoking &nbsp; | &nbsp; 🧘‍♂️ Manage Stress &nbsp; | &nbsp; 🩺 Regular Checkups</span><br><br>
                <ul style='font-size:18px;'>
                  <li>Maintain a balanced diet rich in fruits and vegetables</li>
                  <li>Exercise regularly (at least 30 minutes most days)</li>
                  <li>Avoid smoking and excessive alcohol</li>
                  <li>Manage stress and get regular checkups</li>
                </ul>
            </div>
            """
        else:
            tips_html = """
            <div class='tips-card'>
                <b style='font-size:26px;'>❤️ Tips for Heart Care</b><br><br>
                <span style='font-size:20px;'>🩺 Consult Doctor &nbsp; | &nbsp; 🧑‍⚕️ Monitor Health &nbsp; | &nbsp; 🍎 Heart-Friendly Diet &nbsp; | &nbsp; 🏃‍♂️ Moderate Exercise</span><br><br>
                <ul style='font-size:18px;'>
                  <li>Consult a doctor immediately for proper diagnosis</li>
                  <li>Monitor blood pressure and cholesterol regularly</li>
                  <li>Follow a heart-friendly diet low in saturated fats</li>
                  <li>Exercise moderately only after doctor's advice</li>
                  <li>Take prescribed medications regularly as directed</li>
                </ul>
            </div>
            """
        st.markdown(tips_html, unsafe_allow_html=True)

        # --- PDF Download Section ---
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        st.markdown("### 📄 Download Your Heart Health Report")
        
        # Create PDF with UTF-8 encoding support
        class PDF(FPDF):
            def header(self):
                # Optional: Add header to each page
                pass
            
            def footer(self):
                # Optional: Add footer to each page
                pass

        pdf = PDF()
        pdf.add_page()
        
        # Set font that supports basic characters
        pdf.set_font("Arial", 'B', 20)
        
        # Title
        pdf.cell(200, 10, txt="CardioGuard Heart Health Report", ln=True, align='C')
        pdf.ln(10)
        
        # Patient Info
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Patient Information:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt=f"Name: {st.session_state.prediction_data['name']}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {st.session_state.prediction_data['timestamp']}", ln=True)
        pdf.ln(5)
        
        # Results
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Assessment Results:", ln=True)
        pdf.set_font("Arial", '', 12)
        status = "Healthy" if pred == 0 else "At Risk"
        pdf.cell(200, 10, txt=f"Status: {status}", ln=True)
        pdf.cell(200, 10, txt=f"Heart Disease Risk: {risk_percent}%", ln=True)
        pdf.cell(200, 10, txt=f"Healthy Probability: {healthy_percent}%", ln=True)
        pdf.ln(5)
        
        # Health Parameters
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Health Parameters:", ln=True)
        pdf.set_font("Arial", '', 10)
        
        # Map parameter names to readable labels
        param_labels = {
            'age': 'Age',
            'sex': 'Sex',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Serum Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate Achieved',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'Slope of Peak Exercise ST',
            'ca': 'Number of Major Vessels',
            'thal': 'Thalassemia'
        }
        
        for key, value in st.session_state.prediction_data['input_data'].items():
            param_name = param_labels.get(key, key.replace('_', ' ').title())
            pdf.cell(200, 8, txt=f"{param_name}: {value}", ln=True)
        pdf.ln(5)
        
        # Recommendations (using simple dash instead of bullet point)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Recommendations:", ln=True)
        pdf.set_font("Arial", '', 11)
        
        if pred == 0:
            recommendations = [
                "- Maintain current healthy lifestyle habits",
                "- Continue regular exercise and balanced diet",
                "- Schedule annual heart health checkups",
                "- Monitor blood pressure and cholesterol regularly",
                "- Avoid smoking and excessive alcohol consumption"
            ]
        else:
            recommendations = [
                "- Consult a cardiologist immediately",
                "- Follow prescribed medication regimen",
                "- Adopt heart-healthy diet low in sodium and saturated fats",
                "- Engage in moderate exercise as advised by doctor",
                "- Monitor symptoms and report any changes to your doctor"
            ]
        
        for rec in recommendations:
            pdf.multi_cell(0, 8, txt=rec)
            pdf.ln(2)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 8, txt="Note: This report is generated by CardioGuard AI and should not replace professional medical advice.")
        
        # Generate PDF bytes with proper encoding
        try:
            pdf_bytes = pdf.output(dest='S').encode('latin1')
        except UnicodeEncodeError:
            # Fallback: use simple ASCII encoding
            pdf_bytes = pdf.output(dest='S').encode('ascii', errors='ignore')
        
        # Download button
        st.markdown("<div class='section-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"heart_health_report_{name if name else 'user'}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align:center;font-size:36px;margin-top:2rem;'>✨</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.markdown("#### 👆 Fill in your details and click **Analyze Heart Health** to see your results and download PDF report.")