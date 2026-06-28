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
    text = re.sub(r'<.*?>', '', text)        # remove HTML tags (IMDb has these)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ── Load Product Reviews Dataset ──────────────────────────
print("📦 Loading product reviews dataset...")
df_products = pd.read_csv('dataset/fake_reviews_dataset.csv')
df_products = df_products[['text_', 'label']].copy()
df_products.columns = ['text', 'label']
df_products['label'] = df_products['label'].map({'OR': 0, 'CG': 1})
print(f"✅ Product reviews: {df_products.shape[0]}")

# ── Load IMDb Dataset ─────────────────────────────────────
print("🎬 Loading IMDb dataset...")
df_imdb = pd.read_csv('dataset/IMDB_Dataset.csv')
df_imdb = df_imdb[['review']].copy()
df_imdb.columns = ['text']

# IMDb reviews are all real human reviews = GENUINE (0)
# But to balance — use only a portion to avoid overpowering product data
# We take 10k genuine IMDb reviews to supplement our genuine class
df_imdb_genuine = df_imdb.sample(n=10000, random_state=42).copy()
df_imdb_genuine['label'] = 0  # Genuine
print(f"✅ IMDb reviews sampled: {df_imdb_genuine.shape[0]}")

# ── Combine Datasets ──────────────────────────────────────
print("\n🔀 Combining datasets...")
df_combined = pd.concat([df_products, df_imdb_genuine], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✅ Combined dataset size: {df_combined.shape[0]}")
print(f"Label distribution:\n{df_combined['label'].value_counts()}")

# ── Clean Text ────────────────────────────────────────────
print("\n🧹 Cleaning text (this may take a minute)...")
df_combined['cleaned'] = df_combined['text'].apply(clean_text)

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df_combined['cleaned'], df_combined['label'],
    test_size=0.2, random_state=42, stratify=df_combined['label']
)
print(f"\n📦 Train: {len(X_train)} | Test: {len(X_test)}")

# ── TF-IDF ────────────────────────────────────────────────
print("\n🔢 Vectorizing...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ── Train Models ──────────────────────────────────────────
print("\n🤖 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs')
lr_model.fit(X_train_vec, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_vec))
print(f"✅ Logistic Regression: {lr_acc*100:.2f}%")

print("\n🤖 Training LinearSVC...")
svc_base = LinearSVC(max_iter=2000, C=1.0)
svc_model = CalibratedClassifierCV(svc_base, cv=3)
svc_model.fit(X_train_vec, y_train)
svc_acc = accuracy_score(y_test, svc_model.predict(X_test_vec))
print(f"✅ LinearSVC: {svc_acc*100:.2f}%")

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

# ── Pick Best ─────────────────────────────────────────────
models = {
    'Logistic Regression': (lr_model, lr_acc),
    'LinearSVC': (svc_model, svc_acc),
    'Ensemble': (ensemble, ens_acc)
}
best_name = max(models, key=lambda x: models[x][1])
best_model, best_acc = models[best_name]
best_pred = best_model.predict(X_test_vec)

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

# ── Report ────────────────────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(y_test, best_pred, target_names=['Genuine', 'Fake']))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine','Fake'],
            yticklabels=['Genuine','Fake'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
print("✅ Confusion matrix saved!")

# ── Save ──────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
with open('model/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved!")
print("✅ Vectorizer saved!")
print(f"\n🎉 Done! Best: {best_name} @ {best_acc*100:.2f}%")