# 🧠 Gender-Based Violence Tweet Classification

This project is a machine learning pipeline and web application designed to classify tweets about Gender-Based Violence (GBV) into five categories:

- **Sexual Violence**
- **Emotional Violence**
- **Harmful Traditional Practices**
- **Physical Violence**
- **Economic Violence**

---

## 📁 Dataset

We used two datasets:
- `Train (1).csv` — Contains labeled tweets for training the model.
- `Test (1).csv` — Contains unlabeled tweets for prediction.

Each tweet has:
- `Tweet_ID`
- `tweet` (text content)
- `type` (label for training data only)

---

## 📊 Exploratory Data Analysis (EDA)

Performed the following:
- Checked for duplicates and missing values
- Created bar plots for label distribution
- Generated a WordCloud of most frequent terms

---

## 🧹 Text Preprocessing

1. Convert to lowercase
2. Remove punctuation and URLs
3. Strip HTML tags
4. Tokenize text
5. Remove stopwords
6. Lemmatize words
7. Rejoin tokens into cleaned strings

---


### Feature Extraction:
Used `TfidfVectorizer` with a maximum of 1000 features to vectorize the tweets.

### Models Trained:
- ✅ Random Forest Classifier
- ✅ K-Nearest Neighbors (KNN)
- ✅ Logistic Regression


### How to run
streamlit run app.py

### Acknowledgement
Malebo Phosa


