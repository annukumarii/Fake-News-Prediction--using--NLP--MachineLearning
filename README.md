Fake news has become a major problem in today’s digital world, spreading misinformation across social media and news platforms.
This project builds a Machine Learning model that classifies news articles as:

-- Real News

--Fake News


using Natural Language Processing (NLP) and Logistic Regression.


---

● Tech Stack

Python 3.8+

Pandas, NumPy – Data handling

Scikit-learn – ML models & evaluation

TF-IDF Vectorizer – Text feature extraction

Regex & NLTK – Text preprocessing



---

📊 Dataset

Source: Kaggle Fake News Dataset

Contains news articles with labels:

1 = Fake

0 = Real




---

● How to Run the Project

○ Clone the Repository

○ Install Dependencies

pip install -r requirements.txt

○ Run the Script

python fake_news_detection.py


---

● Project Structure

fake-news-detection/
│── fake_news_detection.py   

│── requirements.txt      

│── README.md         

│── fake_and_real_news.csv


---

● Workflow

1. Data Cleaning

Handle missing values

Remove stopwords, punctuation, numbers

Lemmatization



2. Feature Extraction

TF-IDF Vectorization



3. Modeling

Logistic Regression (best performing model)

Tried Naive Bayes & Random Forest as well



4. Evaluation

Accuracy: ~92%

Metrics: Precision, Recall, F1-score





---

● Example Output

Accuracy: 92%
FAKE NEWS / REAL NEWS Prediction

Test Example:

print(predict_news("Breaking news: Scientists discover water on Mars!"))
# Output → REAL NEWS


---

● Future Improvements

Use Deep Learning models (LSTM, GRU)

Apply BERT/Transformers for better accuracy

Deploy as a Streamlit Web App

