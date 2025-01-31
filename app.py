



# from fastapi import FastAPI
# from fastapi.responses import RedirectResponse
# from pydantic import BaseModel
# import pickle
# from src.preprocess import clean_text

# app = FastAPI()

# # Load trained model and vectorizer
# with open("models/sarcasm_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("models/vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Redirect root URL to Swagger UI
# @app.get("/")
# def redirect_to_docs():
#     return RedirectResponse(url="/docs")

# # Define a request model for input
# class TextRequest(BaseModel):
#     text: str

# @app.post("/predict")
# def predict(data: TextRequest):
#     text = data.text

#     if not text:
#         return {"error": "No text provided"}

#     text_cleaned = clean_text(text)
#     text_tfidf = vectorizer.transform([text_cleaned])
#     prediction = model.predict(text_tfidf)[0]

#     return {"sarcasm": "Sarcastic" if prediction == 1 else "Not Sarcastic"}


from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import pickle
from src.preprocess import clean_text

# Initialize the FastAPI app
app = FastAPI()

# Load trained model and vectorizer
def load_model_and_vectorizer():
    """Load the sarcasm detection model and vectorizer."""
    with open("models/sarcasm_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("models/vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Root endpoint with welcome message and 3-second redirect to /docs
@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <html>
        <head>
            <meta http-equiv="refresh" content="6;url=/docs">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f5f5f5;
                    text-align: center;
                }
                h1 {
                    font-size: 3rem;
                    color: #3f37c9;
                    opacity: 0;
                    animation: fadeIn 2s forwards;
                }
                p {
                    font-size: 1.2rem;
                    color: #888;
                    opacity: 0;
                    animation: fadeIn 3s forwards;
                    animation-delay: 2s;
                }
                @keyframes fadeIn {
                    to {
                        opacity: 1;
                    }
                }
                .redirect {
                    font-size: 1rem;
                    color: grey;
                    opacity: 0;
                    animation: fadeIn 4s forwards;
                    animation-delay: 2s;
                }
            </style>
        </head>
        <body>
            <div>
                <h1>Welcome to the Sarcasm Detection!</h1>
                <p>You will be redirected shortly...</p>
                <p class="redirect">Redirecting to documentation...</p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Define a request model for input
class TextRequest(BaseModel):
    text: str

# Endpoint to predict sarcasm from input text
@app.post("/predict")
def predict(data: TextRequest):
    """Predict if the input text is sarcastic or not."""
    text = data.text

    if not text:
        return {"error": "No text provided"}

    # Clean and transform the text, then make a prediction
    text_cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_tfidf)[0]

    # Return the prediction result
    return {"sarcasm": "Sarcastic" if prediction == 1 else "Not Sarcastic"}
