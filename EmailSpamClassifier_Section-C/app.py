from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')
    
    if not message or not message.strip():
        return render_template('index.html', prediction="⚠️ No message entered.", original="")

    vect = vectorizer.transform([message])
    prediction = model.predict(vect)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template('index.html', prediction=result, original=message)

if __name__ == "__main__":
    app.run(debug=True)
