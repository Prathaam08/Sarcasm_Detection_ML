This project implements a sarcasm detection model using machine learning techniques. It processes text data and predicts whether the given sentence is sarcastic or not. The application provides a web interface where users can enter text and receive predictions along with detailed explanations for the classification.

Features:

Sarcasm Classification: The model predicts if a sentence is sarcastic or not.
Explanations: Provides a detailed explanation for why a sentence is classified as sarcastic or not.
Speech Input: Users can input sentences through speech using voice-to-text (planned feature).
History: View previous predictions (planned feature).
Dark Mode: Toggle between light and dark themes (planned feature).
Clear History: Option to remove past predictions (planned feature).

Requirements
Make sure you have the following dependencies installed:

Python 3.7+
Libraries:
fastapi
uvicorn
scikit-learn
nltk
pandas
pickle

You can install all dependencies by running:
pip install -r requirements.txt

🚀 Setup & Usage

1. Clone this repository:
git clone https://github.com/your-username/sarcasm-detection.git
cd sarcasm-detection

2. Install Dependencies:
pip install -r requirements.txt

3. Dataset
The dataset is expected to be in the Data/fixed_file.json file. You can get the dataset from Kaggle or another source.

4. Train the Model
run : python src/train.py
Once the data is processed, the model is trained using the TF-IDF vectorizer and a classification model (e.g., logistic regression, random forest). The model and vectorizer are saved as .pkl files:
sarcasm_model.pkl
vectorizer.pkl

5. Preprocess Data
The clean_text() function cleans the text data by:
Lowercasing the text
Removing special characters and URLs
Removing stopwords

6. run : python app.py

7. Run the FastAPI Server
To run the server and start serving the sarcasm prediction API, use the following command:
uvicorn app:app --reload

The server will start on http://127.0.0.1:8000.

8. Use the API
Once the server is running, you can send a POST request to /predict with a text input in JSON format.

Example request using curl:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "Oh great, another Monday!"}'

Example response:
{
  "sarcasm": "Sarcastic"
}
