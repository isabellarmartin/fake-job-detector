import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# Load model components
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("kbest_selector.pkl", "rb") as f:
    selector = pickle.load(f)

# App Config
st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.caption("Paste a job description and find out if it's likely **real** or **fake** using our trained model. No cap.")

# Layout columns
col1, col2 = st.columns([2, 1])

with col1:
    job_input = st.text_area("💼 Enter job posting content:", height=250)

with col2:
    st.markdown("### ⚙️ Job Attributes")
    telecommuting = st.checkbox("Telecommuting?", value=False)
    logo = st.checkbox("Has company logo?", value=True)
    questions = st.checkbox("Includes screening questions?", value=False)

# Convert inputs
telecommuting = int(telecommuting)
logo = int(logo)
questions = int(questions)

# Load Random Example
if st.button("🎲 Load Random Job"):
    df = pd.read_csv("fake_job_postings.csv")
    sample = df.sample(1).iloc[0]
    example_text = f"{sample['title']} {sample['description']}"
    st.session_state['example_text'] = example_text

# Pre-fill example if loaded
if 'example_text' in st.session_state:
    st.text_area("🔍 Random Job Example", st.session_state['example_text'], height=200)
    job_input = st.session_state['example_text']

# Prediction function
def predict_post(text, telecommuting, logo, questions):
    text_vec = vectorizer.transform([text])
    numeric_vec = csr_matrix(np.array([[telecommuting, logo, questions]]))
    combined = hstack([text_vec, numeric_vec])
    selected = selector.transform(combined)
    pred = model.predict(selected)[0]
    prob = model.predict_proba(selected)[0][1]
    return pred, prob

# Prediction
if st.button("🔎 Sniff This Job"):
    if not job_input.strip():
        st.warning("Please paste a job description first.")
    else:
        label, prob = predict_post(job_input, telecommuting, logo, questions)

        st.subheader("Prediction Result:")
        if label == 1:
            st.error("🚩 This job is likely **FAKE**.")
            st.caption("🤖 Suspicious vibes detected.")
        else:
            st.success("✅ This job is likely **REAL**.")
            st.caption("📋 Seems legit... but always trust your gut.")

        # Progress bar
        st.progress(prob if label == 1 else 1 - prob)
        st.caption(f"Confidence: {prob:.2%} fake" if label == 1 else f"Confidence: {(1 - prob):.2%} real")

# Footer meme (optional)
st.markdown("---")
st.caption("App built with ❤️ and paranoia about scam jobs.")
