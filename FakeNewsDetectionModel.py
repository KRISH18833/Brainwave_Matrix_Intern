#Build A Fake News Detection Model Using Machine Learning
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset
fake_news = {
    'title': ['Fake News 1', 'Fake News 2'],
    'text': ['This is totally fake news about politics.', 'Another fake news article about health.'],
    'label': [0, 0]
}
real_news = {
    'title': ['Real News 1', 'Real News 2'],
    'text': ['Real news report on economic growth.', 'Genuine news article on healthcare improvements.'],
    'label': [1, 1]
}
fake_df = pd.DataFrame(fake_news)
real_df = pd.DataFrame(real_news)
df = pd.concat([fake_df, real_df]).reset_index(drop=True)
df['text'] = df['title'] + " " + df['text']

# Clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()
df['text'] = df['text'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.5, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))