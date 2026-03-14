import streamlit as st
import pdfplumber
import matplotlib.pyplot as plt
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

st.info("Upload your resume and paste a job description to check how well your resume matches the job requirements.")

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

    # ATS style progress bar
    st.progress(score/100)

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

    # ---------- KEYWORD EXTRACTION ----------
    vectorizer = CountVectorizer(stop_words='english')
    jd_matrix = vectorizer.fit_transform([job_description])

    keywords = vectorizer.get_feature_names_out()

    matched_keywords = []

    for word in keywords:
        if word in resume_text.lower():
            matched_keywords.append(word)

    st.subheader("Matched Keywords from Job Description")

    if matched_keywords:
        st.write(", ".join(matched_keywords[:20]))
    else:
        st.write("No important keywords detected.")

    # ---------- KEYWORD MATCH SCORE ----------
    keyword_score = int((len(matched_keywords) / len(keywords)) * 100)

    st.subheader("Keyword Match Score")
    st.progress(keyword_score/100)
    st.write(f"{keyword_score}% keywords from the job description appear in the resume.")

    # ---------- SKILL MATCH CHART ----------
    st.subheader("Skill Match Visualization")

    skill_counts = [len(resume_skills), len(missing_skills)]
    labels = ["Matched Skills", "Missing Skills"]

    fig, ax = plt.subplots()
    ax.bar(labels, skill_counts)
    st.pyplot(fig)

    # ---------- RESUME SUGGESTIONS ----------
    st.subheader("Resume Suggestions")

    if score < 50:
        st.error("Your resume is missing many required skills.")
    elif score < 70:
        st.warning("Add more relevant skills to improve your ATS score.")
    else:
        st.success("Your resume is well optimized for this job.")

# ---------- FOOTER ----------
st.markdown("""
---
<div class="footer">

👨‍💻 Connect with Me  

🔗 <a href="https://linkedin.com/in/YOUR-LINKEDIN" target="_blank">LinkedIn</a> | 
💻 <a href="https://github.com/YOUR-GITHUB" target="_blank">GitHub</a>  

</div>
""", unsafe_allow_html=True)