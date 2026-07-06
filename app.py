import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "form"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "reasons" not in st.session_state:
    st.session_state.reasons = []

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* ---------- Main background ---------- */
.stApp {
    background: linear-gradient(135deg, #071c2f, #0d3b66, #0f4c75);
    color: white;
}

/* ---------- Layout ---------- */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* ---------- Title ---------- */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.35rem;
}

.sub-text {
    text-align: center;
    font-size: 1.08rem;
    color: #dbeafe;
    margin-bottom: 2rem;
}

/* ---------- Glass cards ---------- */
.card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 26px;
    border-radius: 24px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
    margin-bottom: 18px;
}

.section-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.section-sub {
    font-size: 1.02rem;
    color: #dbeafe;
    margin-bottom: 1.4rem;
}

/* ---------- Inputs ---------- */
label, .stNumberInput label {
    color: white !important;
    font-weight: 700 !important;
}

div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
}

div[data-baseweb="input"] input {
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

input::placeholder {
    color: #d1d5db !important;
}

/* ---------- Attractive Predict Button ---------- */
div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 18px;
    padding: 16px 20px;
    font-size: 1.15rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #6a5cff 100%);
    box-shadow: 0 0 18px rgba(0, 210, 255, 0.35), 0 10px 28px rgba(58,123,213,0.35);
    transition: all 0.25s ease-in-out;
    margin-top: 16px;
}

div.stButton > button:hover {
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 0 28px rgba(0, 210, 255, 0.55), 0 14px 34px rgba(58,123,213,0.45);
}

/* ---------- Secondary buttons ---------- */
.secondary-wrap div.stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    box-shadow: none !important;
}

/* ---------- Tips card ---------- */
.tips-card {
    background: linear-gradient(180deg, rgba(55,220,255,0.16), rgba(255,255,255,0.08));
    border: 1px solid rgba(255,255,255,0.14);
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.22);
    backdrop-filter: blur(10px);
}

.tips-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1rem;
}

.tip-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
}

.tip-head {
    font-size: 1.08rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.35rem;
}

.tip-text {
    color: #e6f3ff;
    line-height: 1.5;
    font-size: 0.98rem;
}

/* ---------- Result boxes ---------- */
.result-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

.approved-box {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(34,197,94,0.28);
}

.rejected-box {
    background: linear-gradient(90deg, #dc2626, #f97316);
    color: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(239,68,68,0.28);
}

.message-text {
    font-size: 1.12rem;
    color: #f8fafc;
    margin-bottom: 1rem;
    font-weight: 600;
}

.hr-line {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.16);
    margin: 20px 0;
}

.summary-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 20px;
}

.summary-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

.summary-item {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    font-size: 1.02rem;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
}

.reason-box {
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #f97316;
    padding: 16px 18px;
    border-radius: 14px;
    margin: 14px 0 18px 0;
}

.reason-title {
    font-size: 1.2rem;
    font-weight: 800;
    margin-bottom: 0.7rem;
    color: #fff;
}

.reason-item {
    color: #fff7ed;
    margin-bottom: 0.45rem;
    font-size: 1rem;
    line-height: 1.45;
}

/* ---------- Hide Streamlit default menu/footer if you want cleaner look ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- VALIDATION + PREDICTION FUNCTION ----------------
def validate_and_predict(age, income, loan_amount, credit_score):
    reasons = []

    # Basic validity / rule checks
    if age < 18 or age > 60:
        reasons.append("Applicant age must be between 18 and 60.")
    if income < 8000:
        reasons.append("Income is too low for loan approval.")
    if loan_amount <= 0:
        reasons.append("Loan amount must be greater than 0.")
    if credit_score < 300 or credit_score > 900:
        reasons.append("Credit score must be between 300 and 900.")

    # Rejection rules
    if loan_amount > income * 0.6:
        reasons.append("Requested loan amount is too high compared to income.")
    if credit_score < 450:
        reasons.append("Credit score is too low for loan approval.")

    # If any rule fails -> reject directly
    if reasons:
        return 0, reasons

    # Model prediction
    sample = np.array([[age, income, loan_amount, credit_score]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]

    if pred == 1:
        return 1, []
    else:
        return 0, ["The model predicted that the application is not eligible."]

# ---------------- FORM PAGE ----------------
def show_form_page():
    st.markdown('<div class="main-title">💳 Loan Approval Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">Enter applicant details below to check whether the loan application is approved or rejected.</div>',
        unsafe_allow_html=True
    )

    # Only one centered form card now
    left_space, center, right_space = st.columns([0.12, 1, 0.12])

    with center:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Applicant Details</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Provide the applicant information below to run the approval prediction.</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=60, value=25, step=1)
            income = st.number_input("Income (₹)", min_value=0, value=50000, step=1000)

        with col2:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=20000, step=1000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=10)

        st.caption("Credit score changes in steps of 10.")

        if st.button("✨ Predict Loan Approval"):
            result, reasons = validate_and_predict(age, income, loan_amount, credit_score)

            st.session_state.prediction_result = result
            st.session_state.reasons = reasons
            st.session_state.input_data = {
                "Age": age,
                "Income": income,
                "Loan Amount": loan_amount,
                "Credit Score": credit_score
            }
            st.session_state.page = "result"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
# ---------------- RESULT PAGE ----------------
def show_result_page():
    data = st.session_state.input_data
    result = st.session_state.prediction_result
    reasons = st.session_state.reasons

    st.markdown('<div class="main-title">💳 Loan Approval Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Application result summary</div>', unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if result == 1:
        st.markdown('<div class="approved-box">✅ Loan Approved</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="message-text">The application is likely to be approved based on the entered details.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="rejected-box">❌ Loan Rejected</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="message-text">The application does not satisfy the eligibility criteria.</div>',
            unsafe_allow_html=True
        )

        if reasons:
            reason_html = '<div class="reason-box"><div class="reason-title">Reason(s)</div>'
            for r in reasons:
                reason_html += f'<div class="reason-item">• {r}</div>'
            reason_html += '</div>'
            st.markdown(reason_html, unsafe_allow_html=True)

    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)

    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">📋 Input Summary</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f'<div class="summary-item"><b>Age:</b> {data["Age"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-item"><b>Income:</b> ₹ {data["Income"]:,}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="summary-item"><b>Loan Amount:</b> ₹ {data["Loan Amount"]:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-item"><b>Credit Score:</b> {data["Credit Score"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_back, col_new = st.columns(2)

    with col_back:
        st.markdown('<div class="secondary-wrap">', unsafe_allow_html=True)
        if st.button("⬅ Back to Form"):
            st.session_state.page = "form"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_new:
        st.markdown('<div class="secondary-wrap">', unsafe_allow_html=True)
        if st.button("🔄 Check Another Application"):
            st.session_state.page = "form"
            st.session_state.prediction_result = None
            st.session_state.reasons = []
            st.session_state.input_data = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE ROUTER ----------------
if st.session_state.page == "form":
    show_form_page()
else:
    show_result_page()