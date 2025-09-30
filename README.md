# Sarcasm Detection Project

## Overview
This project is designed to detect sarcasm in text using machine learning techniques. The model is trained on a sarcasm dataset and utilizes Logistic Regression for classification. The system also considers contextual information such as the situation and the audience to improve accuracy.

## Features
- **Text Preprocessing**: Cleans and prepares text for analysis.
- **Machine Learning Model**: Uses Logistic Regression for classification.
- **User Interface**: Web app where users can input text and get sarcasm predictions.
- **Sarcasm Explanation**: Provides insights into why a sentence was classified as sarcastic.
- **User History**: Save the user history.
- **Dark Mode**: Dark mode .

## Dataset
The project uses a CSV-based dataset, specifically the **Reddit Sarcasm Dataset (SARC)**.

## Model Training
1. **Preprocessing**: Tokenization, lowercasing, stopword removal, and stemming.
2. **Feature Extraction**: TF-IDF vectorization, sentiment scores, and contextual embeddings.
3. **Training**: The Logistic Regression model is trained on the dataset.
4. **Evaluation**: Accuracy, precision, recall, and F1-score are measured.

## Web Application
The sarcasm detection model is integrated into a web application where users can:
- Receive a sarcasm classification (Yes/No) with an explanation.

## Visualizations
The project includes various visualizations to understand sarcasm better:
- **Graph**: Shows sarcastic vs. non-sarcastic sentence count.

## How to Run the Project
### Prerequisites
- Python 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Web App
```bash
python app.py
```
The web application will be available at `http://localhost:5000`.

### Training the Model
```bash
python train.py
```

### Predicting Sarcasm
```bash
python predict.py --text "Oh great, another Monday!"
```

###Future Enhancement 
Future scope of this project is we can enhance the accuracy by providing the confidence percentage determining how much percent the sentence is sarcastic or not .
Afterwards, we can provide situation feild as who us talking to whom which makes a wealthy contribution in predicting sarcasm
