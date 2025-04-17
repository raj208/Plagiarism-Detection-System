import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text


# doc1 = "The quick brown fox jumps over the lazy dog!"
# doc2 = "A quick brown dog outpaces a lazy fox."

# print(preprocess_text(doc1))
# print(preprocess_text(doc2))

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_texts(text_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix, vectorizer


doc1 = "The quick brown fox jumps over the lazy dog!"
doc2 = "A quick brown dog outpaces a lazy fox."

# Preprocess
doc1_clean = preprocess_text(doc1)
doc2_clean = preprocess_text(doc2)

# Vectorize
tfidf_matrix, vectorizer = vectorize_texts([doc1_clean, doc2_clean])

# Check shape of TF-IDF matrix
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
