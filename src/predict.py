import pickle
import sys
from preprocess import clean_text

# Load Model
with open("models/sarcasm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_sarcasm(text):
    text_cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_tfidf)[0]
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"

if __name__ == "__main__":
    user_input = sys.argv[1]  # Take input from command line
    print(predict_sarcasm(user_input))
