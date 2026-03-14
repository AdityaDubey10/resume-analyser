import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Aditya's Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.main-title{
font-size:40px;
font-weight:bold;
text-align:center;
color:#4CAF50;
}

.subtext{
text-align:center;
font-size:18px;
color:gray;
}

.score{
font-size:30px;
font-weight:bold;
color:#4CAF50;
}

.footer{
text-align:center;
font-size:14px;
margin-top:50px;
color:gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AI Resume Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Built by Aditya Dubey</p>', unsafe_allow_html=True)

# ---------- INPUT ----------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")

# ---------- PDF TEXT EXTRACTION ----------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# ---------- MAIN LOGIC ----------
if uploaded_file is not None and job_description != "":

    resume_text = extract_text_from_pdf(uploaded_file)

    text = [resume_text, job_description]

    cv = CountVectorizer()
    matrix = cv.fit_transform(text)

    similarity = cosine_similarity(matrix)[0][1]
    score = round(similarity * 100, 2)

    st.subheader("Match Score")
    st.markdown(f'<p class="score">{score}%</p>', unsafe_allow_html=True)

    if score > 70:
        st.success("Good match for the job!")
    else:
        st.warning("Resume needs improvement.")

    # ---------- SKILLS ----------
    skills = [
        "python","java","sql","machine learning","data analysis",
        "pandas","numpy","django","flask","aws","docker","kubernetes"
    ]

    resume_skills = []

    for skill in skills:
        if skill in resume_text.lower():
            resume_skills.append(skill)

    st.subheader("Detected Skills")

    if resume_skills:
        st.write(", ".join(resume_skills))
    else:
        st.write("No major skills detected.")

    # ---------- MISSING SKILLS ----------
    missing_skills = []

    for skill in skills:
        if skill in job_description.lower() and skill not in resume_text.lower():
            missing_skills.append(skill)

    st.subheader("Suggested Skills to Add")

    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.write("Your resume already matches most skills!")

# ---------- FOOTER ----------
st.markdown("""
---
<div class="footer">

👨‍💻 Connect with Me  

🔗 <a href="https://linkedin.com/in/YOUR-LINKEDIN" target="_blank">LinkedIn</a> | 
💻 <a href="https://github.com/YOUR-GITHUB" target="_blank">GitHub</a>  



</div>
""", unsafe_allow_html=True)