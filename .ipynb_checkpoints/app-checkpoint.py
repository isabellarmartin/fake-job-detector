import streamlit as st
import joblib
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

# Load all models and transformers
model = joblib.load("best_random_forest.pkl")
tfidf_title = joblib.load("tfidf_title.pkl")
tfidf_description = joblib.load("tfidf_description.pkl")
tfidf_requirements = joblib.load("tfidf_requirements.pkl")
tfidf_benefits = joblib.load("tfidf_benefits.pkl")
tfidf_company_profile = joblib.load("tfidf_company_profile.pkl")
kbest_selector = joblib.load("kbest_selector.pkl")
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
st.sidebar.markdown("This tool checks job postings using the text (title, description, etc.) \n" 
                    "**and** signs that important information might be *missing*.")

st.sidebar.markdown("---")

st.sidebar.markdown("#### **Binary Field Explanations:**")
st.sidebar.markdown("- **Title Missing:** No job title provided.\n"
                    "- **Company Profile Missing:** No company profile provided.\n"
                    "- **Description Missing:** No job description provided.\n"
                    "- **Requirements Missing:** No requirements listed.\n"
                    "- **Benefits Missing:** No benefits listed.")

st.sidebar.markdown("(Missing details = fishy behavior 🐟)")

# Form for user input
st.subheader("Fill out the job posting details:")

with st.form("job_form"):
    title_input = st.text_input("Job Title:")
    company_profile_input = st.text_area("Company Profile:")
    description_input = st.text_area("Job Description:")
    requirements_input = st.text_area("Job Requirements:")
    benefits_input = st.text_area("Job Benefits:")

    st.markdown("**Mark any missing fields:**")
    title_missing = st.checkbox("Title is missing", value=False)
    company_profile_missing = st.checkbox("Company Profile is missing", value=False)
    description_missing = st.checkbox("Description is missing", value=False)
    requirements_missing = st.checkbox("Requirements are missing", value=False)
    benefits_missing = st.checkbox("Benefits are missing", value=False)

    st.markdown("**Job Metadata:**")
    telecommuting = st.selectbox("Is it a telecommuting (remote) job?", [0, 1])
    has_company_logo = st.selectbox("Does the job posting have a company logo?", [0, 1])
    has_questions = st.selectbox("Does the job posting have screening questions?", [0, 1])

    submitted = st.form_submit_button("Check the Job Posting!")

if submitted:
    # Transform text fields separately
    X_title = tfidf_title.transform([title_input])
    X_company_profile = tfidf_company_profile.transform([company_profile_input])
    X_description = tfidf_description.transform([description_input])
    X_requirements = tfidf_requirements.transform([requirements_input])
    X_benefits = tfidf_benefits.transform([benefits_input])

    X_text_combined = hstack([X_title, X_company_profile, X_description, X_requirements, X_benefits])

    # Separate metadata and missing fields
    X_meta = np.array([[telecommuting, has_company_logo, has_questions]])
    X_missing = np.array([[title_missing, company_profile_missing, description_missing, requirements_missing, benefits_missing]])

    st.write(f"Scaler expects {scaler.n_features_in_} features.")
    st.write(f"X_meta shape: {X_meta.shape}")

    try:
        X_meta_scaled = scaler.transform(X_meta)
    except Exception as e:
        st.error(f"Scaler transform error: {e}")

    X_meta_sparse = sparse.csr_matrix(X_meta_scaled)
    X_missing_sparse = sparse.csr_matrix(X_missing)

    # Final feature stacking
    X_combined_total = hstack([X_text_combined, X_meta_sparse, X_missing_sparse])

    # Apply feature selection
    X_final = kbest_selector.transform(X_combined_total)

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
        st.success("\n# ✅ This posting looks normal!")
        st.markdown("Always double-check, but this one seems legit based on the details provided.")
        st.image("https://media.giphy.com/media/3orieUe6ejxSFxYCXe/giphy.gif", use_container_width=True)

    st.write("---")

    st.caption("(This tool makes predictions based on patterns — it's not a replacement for your good judgment. If it sounds shady, trust your gut!)")