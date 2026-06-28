import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import re
import pickle
import os

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(w for w in text.split() if w not in sw)

# load product reviews
print("Loading product reviews...")
df1 = pd.read_csv('dataset/fake_reviews_dataset.csv')
df1 = df1[['text_', 'label']].copy()
df1.columns = ['text', 'label']
df1['label'] = df1['label'].map({'OR': 0, 'CG': 1})
print(f"Product reviews loaded: {len(df1)}")

# load imdb reviews (genuine only)
print("Loading IMDb reviews...")
df2 = pd.read_csv('dataset/IMDB_Dataset.csv')[['review']].copy()
df2.columns = ['text']
df2 = df2.sample(n=10000, random_state=42)
df2['label'] = 0
print(f"IMDb reviews sampled: {len(df2)}")

# combine
df = pd.concat([df1, df2], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total: {len(df)} | Labels: {df['label'].value_counts().to_dict()}")

# clean text
print("Cleaning text...")
df['clean'] = df['text'].apply(clean)

# split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# vectorize
print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True, min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# train models
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs')
lr.fit(Xtr, y_train)
print(f"LR Accuracy: {accuracy_score(y_test, lr.predict(Xte))*100:.2f}%")

print("Training LinearSVC...")
svc = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
svc.fit(Xtr, y_train)
print(f"SVC Accuracy: {accuracy_score(y_test, svc.predict(Xte))*100:.2f}%")

print("Training Ensemble...")
ens = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=5.0)),
        ('svc', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3))
    ], voting='soft'
)
ens.fit(Xtr, y_train)
ens_acc = accuracy_score(y_test, ens.predict(Xte))
print(f"Ensemble Accuracy: {ens_acc*100:.2f}%")

# pick best
results = {
    'lr': (lr, accuracy_score(y_test, lr.predict(Xte))),
    'svc': (svc, accuracy_score(y_test, svc.predict(Xte))),
    'ens': (ens, ens_acc)
}
best_key = max(results, key=lambda x: results[x][1])
best_model, best_acc = results[best_key]
best_pred = best_model.predict(Xte)

print(f"\nBest model accuracy: {best_acc*100:.2f}%")
print(classification_report(y_test, best_pred, target_names=['Genuine', 'Fake']))

# confusion matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine','Fake'], yticklabels=['Genuine','Fake'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')

# save model
os.makedirs('model', exist_ok=True)
with open('model/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully.")
