# 🧠 Plagiarism Detection System

A web-based plagiarism detection tool built using **Streamlit**. This application allows users to upload multiple `.txt` or `.pdf` files and checks for textual similarity using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🚀 Features

- ✅ Upload multiple `.txt` and `.pdf` files
- 🔍 Detect similarity between documents using TF-IDF + Cosine Similarity
- 📊 View results in an interactive similarity score table
- ⚡ Simple, fast, and browser-based — no need to install anything beyond Python packages

---

## 🖥️ Demo

![screenshot](screenshot.png) <!-- Add your own screenshot -->

---

## 📁 File Upload Support

- `.txt` — plain text files
- `.pdf` — PDF files (text-extraction supported via PyPDF2)

---

## ⚙️ Technology Stack

- Python
- Streamlit (Web UI)
- scikit-learn (TF-IDF + Cosine Similarity)
- PyPDF2 (PDF text extraction)
- Pandas (tabular display)


## Future Enhancements

- Add NLP-based deep learning models (BERT, etc.)
- Support for More File Formats: Extend support to .docx, .odt, and .html files for broader usability.
- Add graphical views (e.g., heatmaps, bar charts) to visualize overlapping content.
- Plagiarism Threshold Alert: Let users set a similarity threshold and get alerts when it's crossed.


---

## 📦 Installation

```bash
git clone https://github.com/your-username/plagiarism-detector.git
cd plagiarism-detector
pip install -r requirements.txt
streamlit run app.py


