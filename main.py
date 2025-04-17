import streamlit as st
from program3 import clean_text, get_similarity

st.title("📄 Advanced Plagiarism Checker")

# Inputs
text1 = st.text_area("Enter First Text", height=200)
text2 = st.text_area("Enter Second Text", height=200)

# Button to process
if st.button("Check Similarity"):
    if text1 and text2:
        # Clean input
        clean1 = clean_text(text1)
        clean2 = clean_text(text2)

        # Use your model/vectorizer/similarity logic
        similarity_score = get_similarity(clean1, clean2)

        st.write(f"### 🔍 Similarity Score: `{similarity_score:.2f}`")
        if similarity_score > 0.8:
            st.warning("⚠️ High chance of plagiarism!")
        elif similarity_score > 0.5:
            st.info("🤔 Might be partially similar.")
        else:
            st.success("✅ Looks original!")
    else:
        st.error("Please enter both texts to compare.")
