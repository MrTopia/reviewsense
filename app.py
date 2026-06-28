from flask import Flask, render_template, request, session
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = 'reviewsense_secret_123'

# load product model
with open('model/model.pkl', 'rb') as f:
    product_model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    product_vec = pickle.load(f)

# load movie model
with open('model/movie_model.pkl', 'rb') as f:
    movie_model = pickle.load(f)
with open('model/movie_vectorizer.pkl', 'rb') as f:
    movie_vec = pickle.load(f)

sw = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in sw)
    return text

def check_signals(text, platform):
    signals = []
    movie_platform = platform in ['IMDb', 'BookMyShow']

    if text.count('!') > 3:
        signals.append("⚠️ Excessive use of exclamation marks")
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2:
        signals.append("⚠️ Unusual number of ALL CAPS words")

    if movie_platform:
        pos_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                     'incredible', 'outstanding', 'brilliant', 'masterpiece',
                     'greatest', 'superb', 'wonderful', 'awesome']
        common_phrases = ['must watch', 'best movie', 'greatest film', 'best ever',
                          'everyone must', 'life changing', 'best director', 'best actor']
    else:
        pos_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                     'incredible', 'outstanding', 'superb', 'wonderful', 'awesome']
        common_phrases = ['highly recommend', 'must buy', 'five stars', 'best ever',
                          'love it', 'changed my life', 'best purchase', 'not enough stars']

    found_pos = [w for w in pos_words if w in text.lower()]
    if len(found_pos) >= 3:
        signals.append(f"⚠️ Overloaded with positive words: {', '.join(found_pos)}")

    if len(text.split()) < 15:
        signals.append("⚠️ Review is very short — limited information")

    found_phrases = [p for p in common_phrases if p in text.lower()]
    if found_phrases:
        signals.append(f"⚠️ Generic phrases detected: '{found_phrases[0]}'")

    return signals

def get_verdict(conf, label):
    if label == 'FAKE':
        if conf >= 85:
            return "🚨 Very likely fake — Do NOT trust this review", "#ef4444", "rgba(239,68,68,0.15)", "#ef4444"
        elif conf >= 70:
            return "❗ Probably fake — Be very cautious", "#f97316", "rgba(249,115,22,0.15)", "#f97316"
        else:
            return "🤔 Possibly fake — Take with a grain of salt", "#eab308", "rgba(234,179,8,0.15)", "#eab308"
    else:
        if conf >= 85:
            return "✅ Highly trustworthy — Looks genuine", "#22c55e", "rgba(34,197,94,0.15)", "#22c55e"
        elif conf >= 70:
            return "✅ Looks genuine — Fairly trustworthy", "#22c55e", "rgba(34,197,94,0.15)", "#22c55e"
        elif conf >= 50:
            return "🙂 Seems okay — But do your own research", "#84cc16", "rgba(132,204,18,0.15)", "#84cc16"
        else:
            return "😐 Uncertain — Don't rely on this alone", "#eab308", "rgba(234,179,8,0.15)", "#eab308"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    review_text = ''
    platform = 'Amazon'
    opinion = color = bg = border = None
    signals = []
    warning = None
    count = session.get('count', 0)

    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        platform = request.form.get('platform', 'Amazon')

        if len(review_text.split()) < 5:
            warning = "⚠️ Review is too short to analyze accurately. Please enter a longer review."
        else:
            cleaned = preprocess(review_text)

            if platform in ['IMDb', 'BookMyShow']:
                vec = movie_vec
                mdl = movie_model
            else:
                vec = product_vec
                mdl = product_model

            X = vec.transform([cleaned])
            pred = mdl.predict(X)[0]
            prob = mdl.predict_proba(X)[0]
            conf = round(max(prob) * 100, 2)
            result = 'FAKE' if pred == 1 else 'GENUINE'
            signals = check_signals(review_text, platform)

            # rule based override for obvious fakes
            exclaims = review_text.count('!')
            caps = len(re.findall(r'\b[A-Z]{3,}\b', review_text))
            pos_words = ['amazing', 'perfect', 'best', 'excellent', 'fantastic',
                         'incredible', 'outstanding', 'brilliant', 'superb', 'wonderful', 'awesome']
            pos_count = len([w for w in pos_words if w in review_text.lower()])
            gen_phrases = ['highly recommend', 'must buy', 'five stars', 'best ever',
                           'love it', 'changed my life', 'must watch', 'greatest film',
                           'best movie', 'everyone must', 'best director']
            gen_count = len([p for p in gen_phrases if p in review_text.lower()])

            score = 0
            if exclaims > 3: score += 2
            if exclaims > 6: score += 1
            if caps > 2: score += 2
            if pos_count >= 3: score += 2
            if pos_count >= 2: score += 1
            if gen_count >= 1: score += 2

            if score >= 4 and result == 'GENUINE':
                result = 'FAKE'
                conf = min(conf + 15, 95)
                signals.append("🚨 Overridden: Pattern strongly matches fake review behaviour")

            opinion, color, bg, border = get_verdict(conf, result)
            confidence = conf
            count += 1
            session['count'] = count

    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           review_text=review_text,
                           platform=platform,
                           opinion=opinion,
                           color=color,
                           bg=bg,
                           border=border,
                           indicators=signals,
                           warning=warning,
                           total_analyzed=count)

if __name__ == '__main__':
    app.run(debug=True)
