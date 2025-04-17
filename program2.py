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


from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_texts(text_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix, vectorizer



import pandas as pd

# Load the dataset
df = pd.read_csv(r'archive\train_snli.txt', sep='\t', header=None)


# Name the columns
df.columns = ['source_text', 'student_text', 'label']

# Check the data
# print(df.head())



# Check for nulls or issues
# print(df.isnull().sum())

# Remove any rows with missing values (just in case)
df.dropna(inplace=True)

# Convert labels to integers (if they’re not already)
df['label'] = df['label'].astype(int)


import pandas as pd
import re

# Step 1: Load data with column names
df = pd.read_csv(r'archive\train_snli.txt', sep='\t', header=None, names=['sentence1', 'sentence2', 'label'])

# Step 2: Basic text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                           # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)    # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces
    return text


# Step 3: Apply cleaning
df['sentence1'] = df['sentence1'].apply(clean_text)
df['sentence2'] = df['sentence2'].apply(clean_text)

# df.dropna(subset=['sentence1', 'sentence2', 'label'], inplace=True)


# Optional: check sample
# print(df.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Combine both columns for fitting the vectorizer
# all_sentences = pd.concat([df['sentence1'], df['sentence2']])

# Initialize TF-IDF
# tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform
# tfidf.fit(all_sentences)

# Transform sentence1 and sentence2                                                                                                                                    
X1 = tfidf.transform(df['sentence1'])
X2 = tfidf.transform(df['sentence2'])


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity row-wise
similarity_scores = []
for i in range(X1.shape[0]):
    sim = cosine_similarity(X1[i], X2[i])[0][0]
    similarity_scores.append(sim)

# Add to DataFrame
df['similarity_score'] = similarity_scores

# View a few examples
print(df[['sentence1', 'sentence2', 'similarity_score']].head())
