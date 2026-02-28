import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Main App Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1d2b64, #f8cdda);
    background-attachment: fixed;
}

/* Remove default white block */
.main {
    background: transparent;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.12);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

/* Title */
.title-style {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    height: 3.5em;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #ff4b2b;
}

/* High Risk */
.high-risk {
    font-size: 50px;
    font-weight: 900;
    color: #ff1e56;
    text-align: center;
    animation: pulse 1.5s infinite;
}

/* Low Risk */
.low-risk {
    font-size: 50px;
    font-weight: 900;
    color: #00ffae;
    text-align: center;
    text-shadow: 0 0 20px #00ffae;
}

/* Animation */
@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.08);}
    100% {transform: scale(1);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("KNN_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- HEADER ---------------- #
st.markdown("<div class='title-style'>❤️ Heart Disease Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("### AI-powered clinical intelligence system")
st.markdown("---")

# ---------------- INPUT SECTION ---------------- #
st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Analyze Heart Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("<div class='high-risk'>⚠ HIGH RISK DETECTED</div>", unsafe_allow_html=True)
        st.progress(90)
    else:
        st.markdown("<div class='low-risk'>✅ LOW RISK</div>", unsafe_allow_html=True)
        st.progress(40)

    st.markdown("</div>", unsafe_allow_html=True)