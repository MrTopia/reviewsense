import pandas as pd
import numpy as np
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

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def word_count(text):
    return len(str(text).split())

# ── Load IMDb ─────────────────────────────────────────────
print("🎬 Loading IMDb dataset...")
df = pd.read_csv('dataset/IMDB_Dataset.csv')
df['word_count'] = df['review'].apply(word_count)
print(f"✅ Loaded: {df.shape[0]} reviews")

# ── Load Product Fake Reviews ─────────────────────────────
print("📦 Loading product fake reviews for transfer learning...")
df_products = pd.read_csv('dataset/fake_reviews_dataset.csv')
df_fake_products = df_products[df_products['label'] == 'CG'][['text_']].copy()
df_fake_products.columns = ['review']
df_fake_products = df_fake_products.sample(n=5000, random_state=42)
print(f"✅ Fake product reviews sampled: {len(df_fake_products)}")

# ── Strategy ──────────────────────────────────────────────
# GENUINE (label=0):
#   - All negative IMDb reviews (clearly human, critical thinking)
#   - Long positive IMDb reviews (word_count > 100, detailed = human)
# FAKE (label=1):
#   - Short positive IMDb reviews (word_count < 40, vague = suspicious)
#   - Fake product reviews (transfer — similar writing patterns)

df_negative = df[df['sentiment'] == 'negative'].copy()
df_negative['label'] = 0

df_positive = df[df['sentiment'] == 'positive'].copy()
df_long_positive = df_positive[df_positive['word_count'] > 100].copy()
df_long_positive['label'] = 0  # Detailed positive = likely genuine

df_short_positive = df_positive[df_positive['word_count'] < 40].copy()
df_short_positive['label'] = 1  # Vague short positive = suspicious/fake

df_fake_products['label'] = 1  # Fake product reviews = fake

print(f"\n📊 Genuine sources:")
print(f"   Negative reviews: {len(df_negative)}")
print(f"   Long positive reviews: {len(df_long_positive)}")
print(f"\n📊 Fake sources:")
print(f"   Short positive reviews: {len(df_short_positive)}")
print(f"   Fake product reviews: {len(df_fake_products)}")

# ── Balance Dataset ───────────────────────────────────────
df_genuine = pd.concat([
    df_negative[['review', 'label']],
    df_long_positive[['review', 'label']]
], ignore_index=True)

df_fake = pd.concat([
    df_short_positive[['review', 'label']],
    df_fake_products[['review', 'label']]
], ignore_index=True)

# Balance to equal sizes
min_size = min(len(df_genuine), len(df_fake))
df_genuine = df_genuine.sample(n=min_size, random_state=42)
df_fake = df_fake.sample(n=min_size, random_state=42)

df_combined = pd.concat([df_genuine, df_fake], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Final combined dataset: {len(df_combined)}")
print(f"Label distribution:\n{df_combined['label'].value_counts()}")

# ── Clean ─────────────────────────────────────────────────
print("\n🧹 Cleaning text...")
df_combined['cleaned'] = df_combined['review'].apply(clean_text)

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df_combined['cleaned'], df_combined['label'],
    test_size=0.2, random_state=42, stratify=df_combined['label']
)
print(f"\n📦 Train: {len(X_train)} | Test: {len(X_test)}")

# ── Vectorize ─────────────────────────────────────────────
print("\n🔢 Vectorizing...")
movie_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2
)
X_train_vec = movie_vectorizer.fit_transform(X_train)
X_test_vec = movie_vectorizer.transform(X_test)

# ── Train ─────────────────────────────────────────────────
print("\n🤖 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=5.0)
lr.fit(X_train_vec, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_vec))
print(f"✅ LR: {lr_acc*100:.2f}%")

print("\n🤖 Training LinearSVC...")
svc = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
svc.fit(X_train_vec, y_train)
svc_acc = accuracy_score(y_test, svc.predict(X_test_vec))
print(f"✅ SVC: {svc_acc*100:.2f}%")

print("\n🤖 Training Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=5.0)),
        ('svc', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3))
    ],
    voting='soft'
)
ensemble.fit(X_train_vec, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test_vec))
print(f"✅ Ensemble: {ens_acc*100:.2f}%")

# ── Best ──────────────────────────────────────────────────
models = {
    'LR': (lr, lr_acc),
    'SVC': (svc, svc_acc),
    'Ensemble': (ensemble, ens_acc)
}
best_name = max(models, key=lambda x: models[x][1])
best_model, best_acc = models[best_name]
best_pred = best_model.predict(X_test_vec)

print(f"\n🏆 Best: {best_name} @ {best_acc*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, best_pred, target_names=['Genuine', 'Fake']))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Genuine','Fake'],
            yticklabels=['Genuine','Fake'])
plt.title(f'Movie Model - {best_name}')
plt.tight_layout()
plt.savefig('model/movie_confusion_matrix.png')
print("✅ Confusion matrix saved!")

# ── Save ──────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
with open('model/movie_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/movie_vectorizer.pkl', 'wb') as f:
    pickle.dump(movie_vectorizer, f)

print("\n✅ movie_model.pkl saved!")
print("✅ movie_vectorizer.pkl saved!")
print(f"\n🎬 Done! {best_name} @ {best_acc*100:.2f}%")