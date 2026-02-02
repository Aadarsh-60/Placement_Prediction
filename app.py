import streamlit as st
import numpy as np
import pickle
import time

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    class DummyModel:
        def predict(self, data): return [1]
        def predict_proba(self, data): return [[0.1, 0.9]]
    model = DummyModel()

st.set_page_config(page_title="Placement Predictor", layout="wide", page_icon="🎓")


st.markdown("""
<style>
    /* 1. Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* 2. Sidebar Styling (Light Background) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }

    /* Force all text in Sidebar to be Black */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }

    /* 3. Main Title Formatting */
    .main h1 {
        color: #1e3a8a !important;
        text-align: center;
        font-weight: 800;
    }

    /* 4. Main Area Headers & Text */
    h3 {
        color: #000000 !important;
    }
    
    p, label {
        color: #000000 !important;
    }

    /* 5. Input Box Styling */
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ced4da !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* 6. Dialog Text Size */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)


@st.dialog("Prediction Result")
def show_result(prediction, probability):
    if prediction == 1:
        st.markdown(f"""
        <div style="text-align: center; color: #155724; background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb;">
            <h2 style="color: #155724; margin:0;">🎉 Congratulations!</h2>
            <br>
            <p>You have a high chance of getting placed.</p>
            <p><b>Confidence: {probability:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; color: #721c24; background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb;">
            <h2 style="color: #721c24; margin:0;">⚠️ Needs Improvement</h2>
            <br>
            <p>It might be tough, but don't give up!</p>
            <p><b>Placement Probability: {probability:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.title("Placement AI")
    st.write("This tool uses Machine Learning to predict your placement probability.")
    st.markdown("---")
    st.caption("Built with Streamlit & Sklearn")

#  MAIN FORM 
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🎓 Student Placement Prediction</h1>", unsafe_allow_html=True)
st.write("### Enter your details below to get a prediction:")

with st.form("placement_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📚 Academic Details")
        iq = st.number_input("IQ Score", min_value=0.0, max_value=200.0, value=100.0, step=1.0)
        cgpa = st.number_input("CGPA (Cumulative)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        tenth = st.number_input("10th Marks (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        twelfth = st.number_input("12th Marks (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        backlogs = st.selectbox("Current Backlogs", [0, 1, 2, 3, "More than 3"])

    with col2:
        st.markdown("### 💡 Skills & Extras")
        comm_skill = st.number_input("Communication Skills (1-10)", min_value=0, max_value=10, value=5)
        tech = st.number_input("Technical Skills (1-10)", min_value=0, max_value=10, value=5)
        comm = st.number_input("Interview Performance (1-10)", min_value=0, max_value=10, value=5)
        
        c1, c2 = st.columns(2)
        with c1:
            hackathons = st.number_input("Hackathons", min_value=0, value=0)
        with c2:
            certifications = st.number_input("Certifications", min_value=0, value=0)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict My Future", type="primary")


if submitted:
    backlog_val = 4 if backlogs == "More than 3" else backlogs
    
    interview_performance = round((comm_skill + tech) / 2, 1)
    
    with st.spinner("Analyzing..."):
        time.sleep(1)
        
        input_data = np.array([[
            iq, cgpa, tenth, twelfth,
            comm_skill, tech, comm,
            hackathons, certifications, backlog_val,
            interview_performance
        ]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100 

        show_result(prediction, probability)
