import pandas as pd
import numpy as np
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import joblib

class SentimentAnalysis:
    def __init__(self, train_file_path, test_file_path):
        self.trainfilepath = train_file_path
        self.testfilepath = test_file_path
    def load_csv(self, path):# loads csv as a pandas Dataframe. Takes argument path which is string
        dataframe = pd.read_csv(path)
        return dataframe
    def remove_punctuation(self,text): #preprocessing step that removes punctuation from text
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "[", "]", "{", "[]", "{}"))
        return final
    def initialize_tfidf(self, maxdf=0.5): #initializes tfidf with maxdf parameter which is a cutoff for term document frequency
        vectorizer = TfidfVectorizer(decode_error={'ignore'}, stop_words={'english'}, strip_accents='unicode',
                                     dtype=np.float64, max_df= maxdf)
        return vectorizer
    def create_train_test_matrix(self, train, test, vectorizer):#fits vectorizer and creates train and test matrices. Take train/test df's and vectorizer arguments
        fitted_vectorizer = vectorizer.fit(train['text'])
        train_matrix = vectorizer.fit_transform(train['text'])
        test_matrix = vectorizer.transform(test['text'])
        train_label = train['label']
        test_label = test['label']
        return fitted_vectorizer, train_matrix, test_matrix, train_label, test_label
    def sgd_logit_classifier(self, random_state=42, max_iter=10000):# creating logistic regression trained using stochastic gradient descent here
        sgd_logit = SGDClassifier(random_state= random_state, max_iter = max_iter, loss='log')
        return sgd_logit
    def predict(self,model, test_set):# function that gets predictions for a model given the modelname and testset.
        predictions = model.predict(test_set)
        return predictions
    def confusion_matrix(self,predictions, test_labels): #generates a confusionmatrix for each set of predictions
        new = np.asarray(test_labels)
        cm = confusion_matrix(predictions, test_labels)
        return cm
    def start(self):#compiles all steps above and generates confusion matrix. Fits classifier on training set as well.Saves model and vectorizer
        train = self.load_csv(self.trainfilepath)
        test = self.load_csv(self.testfilepath)
        train['text'] = train['text'].apply(self.remove_punctuation)
        vectorizer = self.initialize_tfidf()
        fitted_vectorizer,train_matrix, test_matrix, train_label, test_label = self.create_train_test_matrix(train, test, vectorizer)
        sgd_logit = self.sgd_logit_classifier()
        sgd_logit = sgd_logit.fit(train_matrix,train_label)
        predictions = self.predict(sgd_logit, test_matrix)
        confusion_matrix = self.confusion_matrix(predictions, test_label)
        joblib.dump(sgd_logit, 'sgd_logit_model.pkl')
        joblib.dump(fitted_vectorizer, 'vectorizer.pkl')
        print(confusion_matrix)




