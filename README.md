# Sarcasm Detection Project

## 📌 Overview
This project is designed to detect sarcasm in text using **Logistic regression**. It includes a **web application** where users can input text, select context, and get sarcasm predictions.

## 📂 Dataset
- Uses the **Reddit Sarcasm Dataset (SARC)** in **CSV format**.
- The dataset contains sarcastic and non-sarcastic comments along with metadata.

## ⚙️ Features
✅ **Sarcasm Detection**: Classifies text as **sarcastic** or **non-sarcastic**.
✅ **Sarcasm Explanation**: Provides a brief reasoning behind classification.

## 🔥 Model & Techniques
- **Preprocessing**
  - Removes special characters, URLs, and unnecessary text.
  - Tokenization and word embeddings using **DistilBERT**.
- **Model**: Fine-tuned **DistilBERT** for sarcasm classification.
- **Feature Engineering**
  - **N-grams**, **Sentiment Analysis**, and **POS Tagging**.
  - Context-based understanding from situation and audience.
- **Training**
  - Optimized with **cross-entropy loss** and **Adam optimizer**.

## 📊 Visualizations
- **Graph**: Distribution of sarcastic vs. non-sarcastic comments.

## 🖥️ Web App
- Built using **Flask** (backend) and **HTML/CSS/JS** (frontend).
- Users input text and select the situation before making a prediction.
- Option to submit misclassified sentences for **dataset improvement**.

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sarcasm-detection.git
   cd sarcasm-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   ```
4. Run the web app:
   ```bash
   python app.py
   ```
5. Open your browser and visit **http://127.0.0.1:5000/**.

---
🎯 **Contributions & Feedback**: Open to suggestions and improvements! Submit issues and PRs on GitHub. 🚀

