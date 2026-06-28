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

def wc(text):
    return len(str(text).split())

# load imdb
print("Loading IMDb dataset...")
df = pd.read_csv('dataset/IMDB_Dataset.csv')
df['wc'] = df['review'].apply(wc)
print(f"Loaded: {len(df)} reviews")

# load fake product reviews for transfer learning
print("Loading fake product reviews...")
df_prod = pd.read_csv('dataset/fake_reviews_dataset.csv')
df_fake_prod = df_prod[df_prod['label'] == 'CG'][['text_']].copy()
df_fake_prod.columns = ['review']
df_fake_prod = df_fake_prod.sample(n=5000, random_state=42)
print(f"Fake product reviews: {len(df_fake_prod)}")

# genuine = negative reviews + long positive reviews
df_neg = df[df['sentiment'] == 'negative'].copy()
df_neg['label'] = 0

df_pos = df[df['sentiment'] == 'positive'].copy()
df_long = df_pos[df_pos['wc'] > 100].copy()
df_long['label'] = 0

# fake = short vague positive + fake product reviews
df_short = df_pos[df_pos['wc'] < 40].copy()
df_short['label'] = 1

df_fake_prod['label'] = 1

print(f"Genuine - Negative: {len(df_neg)}, Long positive: {len(df_long)}")
print(f"Fake - Short positive: {len(df_short)}, Fake product: {len(df_fake_prod)}")

# combine and balance
df_genuine = pd.concat([df_neg[['review','label']], df_long[['review','label']]], ignore_index=True)
df_fake = pd.concat([df_short[['review','label']], df_fake_prod[['review','label']]], ignore_index=True)

n = min(len(df_genuine), len(df_fake))
df_genuine = df_genuine.sample(n=n, random_state=42)
df_fake = df_fake.sample(n=n, random_state=42)

df_all = pd.concat([df_genuine, df_fake], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total: {len(df_all)} | {df_all['label'].value_counts().to_dict()}")

# clean
print("Cleaning text...")
df_all['clean'] = df_all['review'].apply(clean)

# split
X_train, X_test, y_train, y_test = train_test_split(
    df_all['clean'], df_all['label'], test_size=0.2, random_state=42, stratify=df_all['label']
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# vectorize
print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True, min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# train
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=5.0)
lr.fit(Xtr, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(Xte))*100:.2f}%")

print("Training LinearSVC...")
svc = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
svc.fit(Xtr, y_train)
print(f"SVC: {accuracy_score(y_test, svc.predict(Xte))*100:.2f}%")

print("Training Ensemble...")
ens = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=5.0)),
        ('svc', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3))
    ], voting='soft'
)
ens.fit(Xtr, y_train)
ens_acc = accuracy_score(y_test, ens.predict(Xte))
print(f"Ensemble: {ens_acc*100:.2f}%")

# best model
all_models = {
    'lr': (lr, accuracy_score(y_test, lr.predict(Xte))),
    'svc': (svc, accuracy_score(y_test, svc.predict(Xte))),
    'ens': (ens, ens_acc)
}
best_key = max(all_models, key=lambda x: all_models[x][1])
best_model, best_acc = all_models[best_key]
best_pred = best_model.predict(Xte)

print(f"\nBest accuracy: {best_acc*100:.2f}%")
print(classification_report(y_test, best_pred, target_names=['Genuine', 'Fake']))

# confusion matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Genuine','Fake'], yticklabels=['Genuine','Fake'])
plt.title('Movie Model - Confusion Matrix')
plt.tight_layout()
plt.savefig('model/movie_confusion_matrix.png')

# save
os.makedirs('model', exist_ok=True)
with open('model/movie_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/movie_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Movie model and vectorizer saved.")
