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

def train_and_save(X_train, X_test, y_train, y_test, tfidf, model_path, vec_path, cm_path, label):
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    print(f"\nTraining {label} models...")

    lr = LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs')
    lr.fit(Xtr, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(Xte))
    print(f"LR: {lr_acc*100:.2f}%")

    svc = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
    svc.fit(Xtr, y_train)
    svc_acc = accuracy_score(y_test, svc.predict(Xte))
    print(f"SVC: {svc_acc*100:.2f}%")

    ens = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=5.0)),
            ('svc', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3))
        ], voting='soft'
    )
    ens.fit(Xtr, y_train)
    ens_acc = accuracy_score(y_test, ens.predict(Xte))
    print(f"Ensemble: {ens_acc*100:.2f}%")

    results = {
        'lr': (lr, lr_acc),
        'svc': (svc, svc_acc),
        'ens': (ens, ens_acc)
    }
    best_key = max(results, key=lambda x: results[x][1])
    best_model, best_acc = results[best_key]
    best_pred = best_model.predict(Xte)

    print(f"\nBest {label} model accuracy: {best_acc*100:.2f}%")
    print(classification_report(y_test, best_pred, target_names=['Genuine', 'Fake']))

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine','Fake'], yticklabels=['Genuine','Fake'])
    plt.title(f'{label} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    os.makedirs('model', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(tfidf, f)

    print(f"{label} model saved to {model_path}")
    return best_acc

# ── PRODUCT MODEL ─────────────────────────────────────────
print("=" * 50)
print("PRODUCT MODEL TRAINING")
print("=" * 50)

print("Loading product reviews...")
df_prod = pd.read_csv('dataset/fake_reviews_dataset.csv')
df_prod = df_prod[['text_', 'label']].copy()
df_prod.columns = ['text', 'label']
df_prod['label'] = df_prod['label'].map({'OR': 0, 'CG': 1})
print(f"Product reviews: {len(df_prod)}")

print("Loading IMDb reviews for genuine supplement...")
df_imdb = pd.read_csv('dataset/IMDB_Dataset.csv')[['review']].copy()
df_imdb.columns = ['text']
df_imdb = df_imdb.sample(n=10000, random_state=42)
df_imdb['label'] = 0
print(f"IMDb reviews sampled: {len(df_imdb)}")

df_p = pd.concat([df_prod, df_imdb], ignore_index=True)
df_p = df_p.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total product dataset: {len(df_p)}")

print("Cleaning text...")
df_p['clean'] = df_p['text'].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df_p['clean'], df_p['label'], test_size=0.2, random_state=42, stratify=df_p['label']
)

product_acc = train_and_save(
    X_train, X_test, y_train, y_test,
    TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True, min_df=2),
    'model/model.pkl', 'model/vectorizer.pkl', 'model/confusion_matrix.png',
    'Product'
)

# ── MOVIE MODEL ───────────────────────────────────────────
print("\n" + "=" * 50)
print("MOVIE MODEL TRAINING")
print("=" * 50)

print("Loading IMDb dataset...")
df_mov = pd.read_csv('dataset/IMDB_Dataset.csv')
df_mov['wc'] = df_mov['review'].apply(wc)
print(f"IMDb loaded: {len(df_mov)}")

print("Loading fake product reviews for transfer learning...")
df_fake_prod = pd.read_csv('dataset/fake_reviews_dataset.csv')
df_fake_prod = df_fake_prod[df_fake_prod['label'] == 'CG'][['text_']].copy()
df_fake_prod.columns = ['review']
df_fake_prod = df_fake_prod.sample(n=5000, random_state=42)
df_fake_prod['label'] = 1
print(f"Fake product reviews: {len(df_fake_prod)}")

df_neg = df_mov[df_mov['sentiment'] == 'negative'][['review']].copy()
df_neg['label'] = 0

df_pos = df_mov[df_mov['sentiment'] == 'positive'].copy()
df_long = df_pos[df_pos['wc'] > 100][['review']].copy()
df_long['label'] = 0
df_short = df_pos[df_pos['wc'] < 40][['review']].copy()
df_short['label'] = 1

df_genuine = pd.concat([df_neg, df_long], ignore_index=True)
df_fake = pd.concat([df_short, df_fake_prod[['review','label']]], ignore_index=True)

n = min(len(df_genuine), len(df_fake))
df_genuine = df_genuine.sample(n=n, random_state=42)
df_fake = df_fake.sample(n=n, random_state=42)

df_m = pd.concat([df_genuine, df_fake], ignore_index=True)
df_m = df_m.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total movie dataset: {len(df_m)}")

print("Cleaning text...")
df_m['clean'] = df_m['review'].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df_m['clean'], df_m['label'], test_size=0.2, random_state=42, stratify=df_m['label']
)

movie_acc = train_and_save(
    X_train, X_test, y_train, y_test,
    TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True, min_df=2),
    'model/movie_model.pkl', 'model/movie_vectorizer.pkl', 'model/movie_confusion_matrix.png',
    'Movie'
)

print("\n" + "=" * 50)
print(f"ALL DONE!")
print(f"Product Model Accuracy : {product_acc*100:.2f}%")
print(f"Movie Model Accuracy   : {movie_acc*100:.2f}%")
print("=" * 50)