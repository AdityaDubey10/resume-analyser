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

.footer{
text-align:center;
font-size:14px;
margin-top:50px;
color:gray;
}

.skill-box{
padding:6px;
border-radius:8px;
margin:4px;
display:inline-block;
background-color:#262730;
color:white;
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

    # ---------- DASHBOARD METRICS ----------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("ATS Score", f"{score}%")

    # ---------- KEYWORD EXTRACTION ----------
    vectorizer = CountVectorizer(stop_words='english')
    jd_matrix = vectorizer.fit_transform([job_description])
    keywords = vectorizer.get_feature_names_out()

    matched_keywords = []

    for word in keywords:
        if word in resume_text.lower():
            matched_keywords.append(word)

    keyword_score = int((len(matched_keywords) / len(keywords)) * 100)

    with col2:
        st.metric("Keyword Match Score", f"{keyword_score}%")

    # ---------- PROGRESS BARS ----------
    st.subheader("ATS Score Progress")
    st.progress(score/100)

    st.subheader("Keyword Match Progress")
    st.progress(keyword_score/100)

    st.divider()

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
        for skill in resume_skills:
            st.markdown(f'<span class="skill-box">✔ {skill}</span>', unsafe_allow_html=True)
    else:
        st.write("No major skills detected.")

    st.divider()

    # ---------- MISSING SKILLS ----------
    missing_skills = []

    for skill in skills:
        if skill in job_description.lower() and skill not in resume_text.lower():
            missing_skills.append(skill)

    st.subheader("Suggested Skills to Add")

    if missing_skills:
        for skill in missing_skills:
            st.warning(f"⚠ {skill}")
    else:
        st.success("Your resume already matches most skills!")

    st.divider()

    # ---------- KEYWORDS ----------
    st.subheader("Matched Keywords from Job Description")

    if matched_keywords:
        st.write(", ".join(matched_keywords[:20]))
    else:
        st.write("No important keywords detected.")

    st.divider()

    # ---------- SKILL MATCH CHART ----------
    st.subheader("Skill Match Visualization")

    skill_counts = [len(resume_skills), len(missing_skills)]
    labels = ["Matched Skills", "Missing Skills"]

    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(labels, skill_counts, color=["#4CAF50","#FF6B6B"])

    ax.set_title("Skill Match Analysis")
    ax.set_ylabel("Number of Skills")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,yval+0.1,yval,ha='center')

    st.pyplot(fig)

    # ---------- PIE CHART ----------
    st.subheader("Skill Distribution")

    fig2, ax2 = plt.subplots()

    ax2.pie(
        skill_counts,
        labels=labels,
        autopct='%1.1f%%',
        colors=["#4CAF50","#FF6B6B"],
        startangle=90
    )

    ax2.axis("equal")

    st.pyplot(fig2)

    st.divider()

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