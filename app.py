import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import PyPDF2
import tempfile
import pandas as pd
import matplotlib

# ----------------- Helper Functions ------------------

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_uploaded_file(file):
    if file.type == "text/plain":
        return str(file.read(), "utf-8")
    elif file.type == "application/pdf":
        return extract_text_from_pdf(file)
    else:
        return None

def compute_similarity(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

# ----------------- Streamlit UI ------------------

st.set_page_config(page_title="Plagiarism Checker", layout="centered")
st.title("📄 Plagiarism Detection System")

uploaded_files = st.file_uploader(
    "Upload multiple .txt or .pdf files to compare:",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) >= 2:
    doc_names = []
    doc_texts = []

    for file in uploaded_files:
        content = read_uploaded_file(file)
        if content:
            doc_names.append(file.name)
            doc_texts.append(content)
        else:
            st.warning(f"❗ Could not read: {file.name}")

    if len(doc_texts) >= 2:
        st.success("✅ Files loaded successfully!")

        # Calculate similarity matrix
        sim_matrix = compute_similarity(doc_texts)

        # Create a DataFrame for results
        df = pd.DataFrame(sim_matrix, columns=doc_names, index=doc_names)
        st.subheader("📊 Similarity Scores")
        st.dataframe(df.style.background_gradient(cmap="YlGnBu", axis=None, vmin=0, vmax=1).format("{:.2f}"))

    else:
        st.warning("You need at least 2 valid documents to compare.")
