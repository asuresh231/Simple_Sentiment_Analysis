from flask import Flask, request
import pandas as pd
import numpy as np
import sklearn
import joblib
app = Flask(__name__)


def load_services():
    ml_model = joblib.load('sgd_logit_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return ml_model, vectorizer
def remove_punctuation(text): #preprocessing step that removes punctuation from text
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "[", "]", "{", "[]", "{}"))
    return final
@app.route('/', methods= ["POST", "GET"])
def analyze_sentiment():
    ml_model, vectorizer = load_services()
    if request.method == "POST":
        movie_review = request.form.get("review")
        movie_review = remove_punctuation([movie_review])
        movie_review = vectorizer.transform([movie_review])
        prediction = ml_model.predict_proba(movie_review)[:,1]
        return ('''
                <h1> Sentiment Analysis of Movie Review is: {}% </h1> 
                <button> <a href='http://127.0.0.1:5000/'>Back</a> </button>
                ''').format(round(100 * prediction.squeeze(), 2))

    return ('''
            <form method="post">
            Input a Movie Review to analyze its sentiment: <br>
            Review: <input type="text" name="review"> <br>
            <input type="submit" value="Submit"> <br>
            </form>
            ''')

def main():
    app.run()

if __name__ == '__main__':
    main()

