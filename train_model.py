import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake = pd.read_csv("fake_news_dataset.csv/Fake.csv")
true = pd.read_csv("fake_news_dataset.csv/True.csv")

# Add labels
fake["label"] = 0   # Fake = 0
true["label"] = 1   # True = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Use only text column
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

score = accuracy_score(y_test, y_pred)
print("Accuracy:", round(score*100, 2), "%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))