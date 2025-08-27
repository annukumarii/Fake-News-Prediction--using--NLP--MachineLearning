# FakeNews.py

import re, string
from html import unescape
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# download
for pkg in ("punkt", "wordnet", "omw-1.4", "stopwords", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(pkg)
    except Exception:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def _pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

# clean
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # remove urls
    text = re.sub(r"<.*?>", " ", text)                # remove html
    text = re.sub(r"\s+", " ", text)                  # collapse spaces
    text = re.sub(r"[^a-z\s']", " ", text)            # keep letters & apostrophe
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tagged = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, _pos(p)) for w, p in tagged]
    return " ".join(lemmas)

# load
df = pd.read_csv("news.csv")  

# merge
if "title" in df.columns:
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["text"] = df["text"].fillna("").astype(str)

# labels
def label_to_int(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    s = s.astype(str).str.lower().str.strip()
    if set(s.unique()).issubset({"fake", "real"}):
        return s.map({"fake": 1, "real": 0})
    if set(s.unique()).issubset({"0", "1"}):
        return s.astype(int)
    return s.map(lambda x: 1 if ("fake" in x or x in {"1", "true", "yes"}) else 0)

y = label_to_int(df["label"])
X = df["text"].apply(preprocess)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# vectorize
tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# eval
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, digits=4))

# save
dump((model, tfidf), "fake_news_model.joblib")
print("Model saved -> fake_news_model.joblib")
