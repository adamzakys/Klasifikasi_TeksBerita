import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')

st.title('Klasifikasi Berita')
st.write('Muhammad Adam Zaky Jiddyansah')

# Load your dataset
df = pd.read_csv('https://raw.githubusercontent.com/adamzakys/SourceFiles/main/Data_Berita_All_Kategori.csv')


# Case Folding
df = df.astype(str)
df["Berita"] = df["Berita"].apply(lambda x: x.lower())

# Tokenizing
def process_tokenize(text):
    text = text.split()
    return text

df["processed_berita"] = df["Berita"].apply(process_tokenize)

# Punctuation Removal
def process_punctuation(tokens):
    cleaned_tokens = [re.sub(r'[.,():-]', '', token) for token in tokens]
    cleaned_tokens = [re.sub(r'\d+', '', token) for token in cleaned_tokens]
    return cleaned_tokens

df['processed_berita'] = df['processed_berita'].apply(process_punctuation)

# Stopword Removal
def process_stopword_token(tokens):
    stop_words = set(stopwords.words("indonesian"))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return " ".join(filtered_tokens)

df['processed_berita'] = df['processed_berita'].apply(process_stopword_token)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_berita'], df['Kategori'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Print the feature names (terms) from TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()


# KNN Classification Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf, y_train)

# User Input
user_input = st.text_area('')

# Preprocess user input
user_input = user_input.lower()
user_input_tokens = process_stopword_token(process_punctuation(process_tokenize(user_input)))

# Transform user input using TF-IDF
user_input_tfidf = tfidf_vectorizer.transform([user_input_tokens])

# Predict the category
prediction = knn_model.predict(user_input_tfidf)

# Display Prediction
st.subheader('Prediction:')
st.write(f'Kategori prediksi artikel berita masukan adalah: {prediction[0]}')

# Evaluate Accuracy
y_pred = knn_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Model Accuracy:')
st.write(f'Akurasi model KNN pada set pengujian adalah: {accuracy:.2f}')

