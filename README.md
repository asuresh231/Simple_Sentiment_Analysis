# Simple_Sentiment_Analysis
For this project I've decided to build a simple sentiment analysis model using the IMDB Movie Reviews Dataset located at https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format. My goal for this project was to build a simple and effective sentiment analysis classifier that can be deployed through a REST API(see below). In terms of labelling, the IMDB Movie Reviews Dataset is already pre-labelled with the label '0' representing a bad sentiment and '1' representing a good sentiment. I included the test set('test.csv') in this repository, but could not include the train set because it was too large. 

## Dependencies
I have included all dependencies and packages that need to be installed in a 'requirements.txt' file 

### Model Building 
The 'SentimentAnalysisModel.py' file in this repository contains the script used to load, preprocess, train and generate predictions for the IMDB reviews. The 'Main.py' file in this repository runs the 'SentimentAnalysisModel.py' file and generates 'sgd_logit_model.pkl','vectorizer.pkl', 'cross_validation_scores.jpg' and 'cm.jpg' files. The 'sgd_logit_model.pkl' file is the Stochastic Gradient Descent Logistic Regression model I saved using the joblib library to conduct sentiment analysis. The 'vectorizer.pkl' file is my tfidf embedder saved using joblib. The 'cross_validation_scores.jpg' file is an image of the plot I created to measure the effectiveness of my model using cross validation. For this project, I used k-fold cross validation with 5 k-folds. Finally, the 'cm.jpg' file is the confusion matrix generated from the predictions my SDG-Logistic Regression model made on the test set.



#### Flask Application
To deploy, I decided to create a Flask application. The application is located in the 'flask_app.py' file. This app when run is simply a microservice that takes a review as an argument and returns the probability that the review is positive. 
