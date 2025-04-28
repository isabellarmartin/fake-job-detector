import streamlit as st
import joblib
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

# Load all models and transformers
model = joblib.load("best_random_forest.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
kbest_selector = joblib.load("kbest_selector.pkl")
scaler = joblib.load("scaler.pkl")

import joblib
joblib.dump(selector, "kbest_selector.pkl")


# Set page config
st.set_page_config(page_title="Fake Job Detector", page_icon="🌍", layout="centered")

# Title
st.title("Fake Job Posting Detector")
st.markdown("### Check to see if a job posting might be fake")

st.write("---")

# Sidebar fun
st.sidebar.image("https://media.giphy.com/media/26gsiCIKW7ANEmxKE/giphy.gif", use_container_width=True)
st.sidebar.markdown("#### **How it works:**")
st.sidebar.markdown("This tool checks job postings using the text (title, description, etc.) \n" 
                    "**and** information about the job (like if important fields are missing).")

st.sidebar.markdown("---")

st.sidebar.markdown("#### **Binary Field Explanations:**")
st.sidebar.markdown("- **Telecommuting:** Whether it's a remote job (1 = yes, 0 = no).\n"
                    "- **Company Logo Present:** Whether the posting has a company logo.\n"
                    "- **Questions Asked:** Whether the posting asks additional screening questions.")

st.sidebar.markdown("(Missing details? Hmmm seems fishy. )")

# Form for user input
st.subheader("Fill out the job posting details:")

with st.form("job_form"):
    title_input = st.text_input("Job Title:")
    description_input = st.text_area("Job Description:")
    requirements_input = st.text_area("Job Requirements:")
    benefits_input = st.text_area("Job Benefits:")
    company_profile_input = st.text_area("Company Profile:")

    st.markdown("**Job Metadata:**")
    telecommuting = st.selectbox("Is it a telecommuting (remote) job?", [0, 1])
    has_company_logo = st.selectbox("Does the job posting have a company logo?", [0, 1])
    has_questions = st.selectbox("Does the job posting have screening questions?", [0, 1])

    submitted = st.form_submit_button("Check the Job Posting!")

if submitted:
    # Combine text fields
    combined_text = " ".join([title_input, company_profile_input, description_input, requirements_input, benefits_input])

    # Transform the combined text
    X_text = tfidf_vectorizer.transform([combined_text])

    # Numeric features
    X_numeric = np.array([[telecommuting, has_company_logo, has_questions]])
    X_numeric_scaled = scaler.transform(X_numeric)
    X_numeric_sparse = sparse.csr_matrix(X_numeric_scaled)

    # Combine text and numeric features
    X_combined = hstack([X_text, X_numeric_sparse])

    # Apply feature selection
    X_final = kbest_selector.transform(X_combined)

    # Predict
    probs = model.predict_proba(X_final)[:, 1]
    prediction = (probs >= 0.4).astype(int)

    st.write("---")
    st.subheader("Results:")

    if prediction[0] == 1:
        st.error("\n# ⚡ ALERT: This posting looks suspicious!")
        st.markdown("Beware of jobs with missing details, vague descriptions, or promises that sound too good to be true.\n"
                    "Scammers love lazy job listings.")
        st.image("https://media.giphy.com/media/3o7aCSPqXE5C6T8tBC/giphy.gif", use_container_width=True)
    else:
        st.success("\n# This posting looks normal!")
        st.markdown("Always double-check, but this one seems legit based on the details provided.")
        st.image("https://media.giphy.com/media/3orieUe6ejxSFxYCXe/giphy.gif", use_container_width=True)

    st.write("---")

    st.caption("(This tool makes predictions based on patterns \u2014 it's not a replacement for your good judgment. If it sounds shady, trust your gut!)")