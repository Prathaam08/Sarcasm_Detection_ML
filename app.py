

from flask import Flask, render_template, request, jsonify
import pickle
from src.preprocess import clean_text

app = Flask(__name__)

# Load trained model and vectorizer
with open("models/sarcasm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def explain_sarcasm(sentence, prediction):
    """Generate a detailed explanation for why a sentence is sarcastic or not."""
    
    words = sentence.lower().split()
    sarcasm_indicators = ["oh", "great", "amazing", "wonderful", "yeah", "right", "sure", "totally", "perfect", "love"]
    negation_words = ["not", "never", "no", "barely", "hardly", "scarcely"]
    
    highlighted_words = [word for word in words if word in sarcasm_indicators]
    negation_present = any(word in words for word in negation_words)

    if prediction == 1:  # If sarcastic
        if highlighted_words and negation_present:
            explanation = f"This sentence is sarcastic because it contains **positive words** ({', '.join(highlighted_words)}) **combined with negation**, creating a contradictory tone."
        elif highlighted_words:
            explanation = f"This sentence is sarcastic because it contains **words often used sarcastically**: {', '.join(highlighted_words)}."
        else:
            explanation = "This sentence is sarcastic due to its **tone and phrasing**, which often indicate sarcasm."
    
    else:  # If not sarcastic
        explanation = "This sentence is not sarcastic because it lacks exaggeration, contradiction, or common sarcasm indicators."

    return explanation


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sentence = data.get("text", "")

    if not sentence:
        return jsonify({"sarcasm": "Error: No text provided"}), 400

    cleaned_text = clean_text(sentence)
    text_vectorized = vectorizer.transform([cleaned_text])
    pred = model.predict(text_vectorized)[0]

    result = "Sarcastic" if pred == 1 else "Not Sarcastic"
    explanation = explain_sarcasm(sentence, pred)

    return jsonify({"sarcasm": result, "explanation": explanation})

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=8000)

