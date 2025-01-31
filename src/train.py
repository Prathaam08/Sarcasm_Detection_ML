import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import load_data

X_train, X_test, y_train, y_test, vectorizer = load_data()

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save Model and Vectorizer
with open("models/sarcasm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
