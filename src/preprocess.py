import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def clean_text(text):
    """Preprocess text: lowercase, remove special characters, stopwords"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def load_data(filepath="Data/fixed_file_v2.json"):
    """Load and preprocess the dataset"""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load JSON file

    df = pd.DataFrame(data)
    df = df[['headline', 'is_sarcastic']]  # Keep only relevant columns
    df['text'] = df['headline'].apply(clean_text)  # Preprocess text

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_sarcastic'], test_size=0.2, random_state=42)

    # Convert text to numerical format
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Quick test: Load and check data
if __name__ == "__main__":
    df_test = load_data()
    print("Dataset loaded successfully!")
