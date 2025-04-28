import streamlit as st
import joblib
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

# Load your model
model = joblib.load("best_random_forest.pkl")

# Load your saved TF-IDF vectorizers and scaler
tfidf_title = joblib.load("tfidf_title.pkl")
tfidf_description = joblib.load("tfidf_description.pkl")
tfidf_requirements = joblib.load("tfidf_requirements.pkl")
tfidf_benefits = joblib.load("tfidf_benefits.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="Fake Job Detector", page_icon="🌍", layout="centered")

# Title
st.title("Fake Job Posting Detector")
st.markdown("### Check to see if a job posting might be fake")

st.write("---")

# Sidebar fun
st.sidebar.image("https://media.giphy.com/media/26gsiCIKW7ANEmxKE/giphy.gif", use_container_width=True)
st.sidebar.markdown("#### **How it works:**")
st.sidebar.markdown("This tool checks job postings using both the text (like title and description) \n" 
                    "**and** signs that important information might be *missing*.")

st.sidebar.markdown("---")

st.sidebar.markdown("#### **Binary Field Explanations:**")
st.sidebar.markdown("- **Title Missing:** The job listing has no title.\n"
                    "- **Company Profile Missing:** No company description was provided.\n"
                    "- **Description Missing:** The job description is empty.\n"
                    "- **Requirements Missing:** No skills/qualifications listed.\n"
                    "- **Benefits Missing:** No benefits were mentioned.")

st.sidebar.markdown("(Missing fields? Hmmm this seems fishy)")

# Form for user input
st.subheader("Fill out the job posting details:")

with st.form("job_form"):
    title_input = st.text_input("Job Title:")
    description_input = st.text_area("Job Description:")
    requirements_input = st.text_area("Job Requirements:")
    benefits_input = st.text_area("Job Benefits:")
    company_profile_input = st.text_area("Company Profile (optional):")

    st.markdown("**Check if any fields are missing:**")
    title_missing = st.checkbox("Title is missing", value=False)
    company_profile_missing = st.checkbox("Company profile is missing", value=False)
    description_missing = st.checkbox("Description is missing", value=False)
    requirements_missing = st.checkbox("Requirements are missing", value=False)
    benefits_missing = st.checkbox("Benefits are missing", value=False)

    submitted = st.form_submit_button("Check the Job Posting!")

if submitted:
    # Transform text fields
    X_title = tfidf_title.transform([title_input])
    X_description = tfidf_description.transform([description_input])
    X_requirements = tfidf_requirements.transform([requirements_input])
    X_benefits = tfidf_benefits.transform([benefits_input])

    X_text_combined = hstack([X_title, X_description, X_requirements, X_benefits])

    # Numeric missing indicators
    X_numeric = np.array([[title_missing, company_profile_missing, description_missing, requirements_missing, benefits_missing]])
    X_numeric_scaled = scaler.transform(X_numeric)
    X_numeric_sparse = sparse.csr_matrix(X_numeric_scaled)

    # Final feature stacking
    X_final = hstack([X_text_combined, X_numeric_sparse])

    # Predict probability and apply threshold
    probs = model.predict_proba(X_final)[:, 1]
    prediction = (probs >= 0.4).astype(int)

    st.write("---")
    st.subheader("Results:")

    if prediction[0] == 1:
        st.error("\n# ALERT: This posting looks suspicious!")
        st.markdown("Beware of jobs with missing details, vague descriptions, or promises that sound too good to be true.\n"
                    "Scammers love lazy job listings.")
        st.image("https://media.giphy.com/media/3o7aCSPqXE5C6T8tBC/giphy.gif", use_container_width=True)
    else:
        st.success("\n# This posting looks normal!")
        st.markdown("Always double-check, but this one seems legit based on the details provided.")
        st.image("https://media.giphy.com/media/3orieUe6ejxSFxYCXe/giphy.gif", use_container_width=True)

    st.write("---")

    st.caption("(This tool makes predictions based on patterns \u2014 it's not a replacement for your good judgment. If it sounds shady, trust your gut!)")
