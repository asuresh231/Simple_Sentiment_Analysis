import numpy as np
import SentimentAnalysisModel
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import joblib
from SentimentAnalysisModel import SentimentAnalysis


def main():
    sentiment_analysis = SentimentAnalysis('Train.csv', 'Test.csv')
    sentiment_analysis.start()


if __name__ == '__main__':
    main()
