from flask import Flask, render_template, request, session
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import nltk
import os

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = 'fakereviewdetector2024'

# ── Load Models ───────────────────────────────────────────
# Product model (Amazon, Flipkart)
with open('model/model.pkl', 'rb') as f:
    product_model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    product_vectorizer = pickle.load(f)

# Movie model (IMDb, BookMyShow)
with open('model/movie_model.pkl', 'rb') as f:
    movie_model = pickle.load(f)
with open('model/movie_vectorizer.pkl', 'rb') as f:
    movie_vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def get_fake_indicators(text, platform='Amazon'):
    indicators = []
    is_movie = platform in ['IMDb', 'BookMyShow']

    if text.count('!') > 3:
        indicators.append("⚠️ Excessive use of exclamation marks")
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2:
        indicators.append("⚠️ Unusual number of ALL CAPS words")

    if is_movie:
        positive_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                          'incredible', 'outstanding', 'brilliant', 'masterpiece',
                          'greatest', 'superb', 'wonderful', 'awesome']
        generic = ['must watch', 'best movie', 'greatest film', 'best ever',
                   'everyone must', 'life changing', 'best director', 'best actor']
    else:
        positive_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                          'incredible', 'outstanding', 'superb', 'wonderful', 'awesome']
        generic = ['highly recommend', 'must buy', 'five stars', 'best ever',
                   'love it', 'changed my life', 'best purchase', 'not enough stars']

    found = [w for w in positive_words if w in text.lower()]
    if len(found) >= 3:
        indicators.append(f"⚠️ Overloaded with positive words: {', '.join(found)}")

    if len(text.split()) < 15:
        indicators.append("⚠️ Review is very short — limited information")

    found_generic = [p for p in generic if p in text.lower()]
    if found_generic:
        indicators.append(f"⚠️ Generic phrases detected: '{found_generic[0]}'")

    return indicators

def get_confidence_info(confidence, result):
    if result == 'FAKE':
        if confidence >= 85:
            opinion = "🚨 Very likely fake — Do NOT trust this review"
            color = "#ef4444"
            bg = "rgba(239,68,68,0.15)"
            border = "#ef4444"
        elif confidence >= 70:
            opinion = "❗ Probably fake — Be very cautious"
            color = "#f97316"
            bg = "rgba(249,115,22,0.15)"
            border = "#f97316"
        else:
            opinion = "🤔 Possibly fake — Take with a grain of salt"
            color = "#eab308"
            bg = "rgba(234,179,8,0.15)"
            border = "#eab308"
    else:
        if confidence >= 85:
            opinion = "✅ Highly trustworthy — Looks genuine"
            color = "#22c55e"
            bg = "rgba(34,197,94,0.15)"
            border = "#22c55e"
        elif confidence >= 70:
            opinion = "✅ Looks genuine — Fairly trustworthy"
            color = "#22c55e"
            bg = "rgba(34,197,94,0.15)"
            border = "#22c55e"
        elif confidence >= 50:
            opinion = "🙂 Seems okay — But do your own research"
            color = "#84cc16"
            bg = "rgba(132,204,18,0.15)"
            border = "#84cc16"
        else:
            opinion = "😐 Uncertain — Don't rely on this alone"
            color = "#eab308"
            bg = "rgba(234,179,8,0.15)"
            border = "#eab308"

    return confidence, opinion, color, bg, border

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    review_text = ''
    platform = 'Amazon'
    opinion = None
    color = None
    bg = None
    border = None
    indicators = []
    warning = None
    total_analyzed = session.get('total_analyzed', 0)

    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        platform = request.form.get('platform', 'Amazon')

        if len(review_text.split()) < 5:
            warning = "⚠️ Review is too short to analyze accurately. Please enter a longer review."
        else:
            cleaned = clean_text(review_text)

            # ── Select model based on platform ────────────
            if platform in ['IMDb', 'BookMyShow']:
                active_model = movie_model
                active_vectorizer = movie_vectorizer
            else:
                active_model = product_model
                active_vectorizer = product_vectorizer

            vectorized = active_vectorizer.transform([cleaned])
            prediction = active_model.predict(vectorized)[0]
            proba = active_model.predict_proba(vectorized)[0]
            raw_confidence = round(max(proba) * 100, 2)
            result = 'FAKE' if prediction == 1 else 'GENUINE'
            indicators = get_fake_indicators(review_text, platform)

            # ── Rule based override ───────────────────────
            caps_count = len(re.findall(r'\b[A-Z]{3,}\b', review_text))
            exclaim_count = review_text.count('!')
            if platform in ['IMDb', 'BookMyShow']:
                positive_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                                  'incredible', 'outstanding', 'brilliant', 'masterpiece',
                                  'greatest', 'superb', 'wonderful', 'awesome']
            else:
                positive_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                                  'incredible', 'outstanding', 'superb', 'wonderful', 'awesome']

            positive_count = len([w for w in positive_words if w in review_text.lower()])
            generic = ['highly recommend', 'must buy', 'five stars', 'best ever',
                       'love it', 'changed my life', 'best purchase', 'must watch',
                       'greatest film', 'best movie', 'everyone must', 'best director']
            generic_count = len([p for p in generic if p in review_text.lower()])

            override_score = 0
            if exclaim_count > 3: override_score += 2
            if exclaim_count > 6: override_score += 1
            if caps_count > 2: override_score += 2
            if positive_count >= 3: override_score += 2
            if positive_count >= 2: override_score += 1
            if generic_count >= 1: override_score += 2

            if override_score >= 4 and result == 'GENUINE':
                result = 'FAKE'
                raw_confidence = min(raw_confidence + 15, 95)
                indicators.append("🚨 Overridden: Pattern strongly matches fake review behaviour")

            confidence, opinion, color, bg, border = get_confidence_info(raw_confidence, result)

            total_analyzed += 1
            session['total_analyzed'] = total_analyzed

    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           review_text=review_text,
                           platform=platform,
                           opinion=opinion,
                           color=color,
                           bg=bg,
                           border=border,
                           indicators=indicators,
                           warning=warning,
                           total_analyzed=total_analyzed)

if __name__ == '__main__':
    app.run(debug=True)