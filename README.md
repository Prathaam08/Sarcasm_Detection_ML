This project is a Sarcasm Detection System built using machine learning, where a model is trained to classify whether a given text (headline) is sarcastic or not. The backend of the project is powered by FastAPI to expose a prediction API.

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



🗂️ Folder Structure
sarcasm-detection/
│
├── Data/
│   └── fixed_file_v2.json  # Your dataset
│
├── models/
│   ├── sarcasm_model.pkl   # Trained model
│   └── vectorizer.pkl      # TF-IDF vectorizer
│
├── src/
│   ├── fix_json.py         # Script to fix or preprocess the dataset (if needed)
│   ├── predict.py          # API prediction logic (handles text prediction)
│   ├── preprocess.py       # Functions for text cleaning
│   ├── train.py            # Script to train the model
│
├── app.py                  # FastAPI app
├── README.md               # This File
└── requirements.txt        # Project dependencies

