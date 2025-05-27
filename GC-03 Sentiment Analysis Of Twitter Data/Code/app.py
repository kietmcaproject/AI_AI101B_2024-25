from flask import Flask, render_template, request, url_for
from sentiment_model import analyze_sentiment
from utils import clean_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    tweet = '' 
    
    if request.method == 'POST':
        tweet = request.form.get('tweet', '')
        cleaned = clean_text(tweet)
        sentiment, confidence = analyze_sentiment(cleaned)
        confidence = round(confidence * 100, 2)

    return render_template('index.html', sentiment=sentiment, confidence=confidence, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
