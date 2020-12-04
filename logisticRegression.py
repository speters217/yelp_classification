"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 12/3/20
Logistic Regression Classifier for Yelp Dataset

Background:
This program aims to create a model that maximizes root mean squared error 
on a dataset of Yelp reviews from restaurants near Madison, WI. The final model 
placed first in my UW-Madison Stat 333 class' Kaggle competition.

The Data:
We were given 57,008 reviews that consisted of the star rating, text, city, 
postal code, number of words and characters, number of 'positive' and 
'negative words', as well as term frequency for 500 words of the review. We 
were also given a similar dataset of 38,005 entries for which we had to predict 
the star ratings. I decided to preprossess the text on my own using the text of 
the reviews, rather than using the provided predictors.

My Process:
First, I (naively) started by creating my own implementation of a Naive Bayes 
classifier using a bag of words approach. At its best it could achieve an 
rmse of 1.2 while taking a good deal of time. To its credit, it did only 
classify discrete 1-5 values which was not the best for rmse.

Next, I looked to Scikit-learn's repertoire of model creation software. I 
started by vectorizing the text into a tf-idf matrix. Then, I split the 
data into a 75:25 training:testing ratio. Lastly, I implemented SMOTE oversampling. 
I would spend quite a while trying out various models with various 
parameters. After I found the best parameters for the tf-idf encoding, here 
are the rmse scores for the best versions I was able to achieve for most of 
the models I tried (though I abandoned many after finding success with Logistic Regression):

Multinomial Naive Bayes: 0.8115149239086666
XGBoost Classifier: 0.7817620124917976
Logistic Regression: 0.6694512778569809
    
The Final Model:
    Type: Logistic Regression
    Tokens: 1-grams through 5-grams
    Document Frequency: 4 documents - 50% of documents
    Resampling: SMOTE 'not minority'
    C Inverse Regularization: 5
    Penalty: elaticnet of 0.5
    Max # iterations to converge: 1000
    Time to build: 1:03:36
    25% test RMSE: 0.6694512778569809
    Kaggle Public RMSE: 0.66942
    Kaggle Private RMSE: 0.66440

Final Remarks: 
While the model performs well, it could still do with some improvement.
For one, the model parameters could be tuned with more accuracy, though I argue
with diminishing returns. Tuning the regularization strength on smaller values 
(larger C) could improve the score, though an increase from 4 to 5 saw small gains and 
roughly 3 times the running time. Also, I was unable to give the solver and 
penalty combinations sufficient attention. I also think that tree-based models 
such as XGBoost and Random Forest have promise, yet I didn't have enough time 
or resources to explore those options sufficiently. After I created the final 
model, I adjusted the code so that the testing set was not used for tf-idf 
vectorization. This makes the code more modular and easier to predict on any 
set, though it changes the model results slightly.
"""

import numpy as np
import pandas as pd
from pickle import load, dump
import time

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# My own functions
from lib.load_files import get_files
from lib.print_info import print_welcome_text, convert_elapsed_time
from lib.vectorize_text import vectorize_text
from lib.predict import predict
from lib.visualization import visualize

# We want to ignore any convergence warnings
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')


############################ Vocabulary Parameters ###########################
# Cutoff where we don't accept tokens
# An integer correlates to number of documents that the token appeared in
# Float (0<x<1) correlates to ratio of the documents that contain the token.
MIN_DF = 4 # Lower limit
MAX_DF = 0.5 # Upper limit
# 1.0 means no cutoff

# Vocabulary will only have the top max_features ordered by term frequency across the corpus.
MAX_FEATURE = None

MIN_GRAM = 1 # Min n-gram
MAX_GRAM = 5 # Max n-gram


####################### Logistic Regression Parameters #######################

MODEL_TYPE = 'LogReg'

# Specifies the normalization used for the penalty.
PENALTY = 'elasticnet'
# Elastic-Net mixing paramater. Float between 0 and 1 where 0 and 1 are 
# l2 and l1 penalties, respectively.
L1_RATIO = 0.5 

# The maximum number of iterations to converge.
MAX_ITER = 1000

# Inverse of regularization strength.
C = 5

# Algorithm to use in optimaztion problem.
SOLVER = 'saga'

# Percent of training data to become testing data.
TEST_RATIO = 0.25

# Resampling strategy to use with SMOTE.
SAMPLE_STRAT = 'not minority'

################################# File Paths #################################
# Note that all csvs should be produced from parser.py

# True if a model should be loaded.
# Note that if a model is to be loaded, the program checks for the
# corresponding tf-idf file in the VECTOR directory.
# Settings above should match that of the model.
LOAD_MODEL = False
# Name of model to load
MODEL_NAME = 'best_model.pickle'
USE_MINI = False # Control whether the mini training set is used

# Load in all of the files needed
TRAIN_CSV, VERSION, TEST_CSV, PREDICTION_CSV, VECTOR, X_TRAIN, MODEL, MODEL_IN, LOAD \
= get_files(USE_MINI, MIN_GRAM, MAX_GRAM, MIN_DF, MAX_DF, LOAD_MODEL, MODEL_NAME, 
            MODEL_TYPE, SOLVER, C)


"""
Trains a model using Logistic Regression. First splits the data into a 
25:75 testing:training ration. Then, SMOTE oversampling is performed on the 
training data. The model is then fit and the rmse of the training and testing 
sets are printed. Lastly the model is saved and returned.

INPUT:
x_train_original: A TF-IDF matrix for the training set.
y_train_original: String labels for the training set.
model_name: Filepath of location to save the best trained model to.
penalty: Penalty to be used for the model.
max_iter: The maximum number of iterations to perform for convergence.
l1_ratio: Elastic-Net mixing paramater. Float between 0 and 1 where 0 and 1 are 
        l2 and l1 penalties, respectively
max_iter: The maximum number of iterations to converge.
c: Inverse of regularization strength.
solver: Algorithm to use in optimaztion problem.
test_ratio: Percent of training data to become testing data.
sample_strat: Resampling strategy to use with SMOTE.

OUTPUT:
Saves and returns the model constructed from the given data and parameters.
"""
def train(x_train_original, y_train_original, model_name, penalty, l1_ratio, 
          max_iter, c, solver, test_ratio, sample_strat):    
    # Split the data into 75% training, 25% testing
    x_train_old, x_test, y_train_old, y_test = train_test_split(x_train_original, 
                                                                y_train_original, 
                                                                test_size = test_ratio)

    # Use SMOTE to synthetically resample using sample_strat
    print("Sampling strategy: {}".format(sample_strat))
    sampler = SMOTE(sampling_strategy=sample_strat)
    x_train, y_train = sampler.fit_sample(x_train_old, y_train_old)        

    # The Logistic Regression model that will be fit on the training set
    logReg = LogisticRegression(penalty=penalty, max_iter=max_iter, C=c,
                                solver=solver, l1_ratio=l1_ratio, n_jobs=-1)
    
    # Fit the training data to the model
    logReg.fit(x_train, y_train)
    
    # Get and print rmse scores of the testing and training sets
    score_test = calculate_rmse(logReg, x_test, y_test)
    score_train = calculate_rmse(logReg, x_train, y_train)
    print("Rmse of TRAINING set: {}".format(score_train))
    print("Rmse of TESTING set: {}".format(score_test))
    
    # Save and return the model
    dump(logReg, open(model_name.format(str(score_test).replace(".", "_")), 'wb'))
    return logReg


"""
Calculates and returns the rmse.

INPUT:
estimator: The logistic regression estimator.
x: The input that the estimator will use to predict.
y: The actual labels that we will compare the predictions to.

OUTPUT:
returns the rmse of the x as predicted by estimator and compared to y.
"""
def calculate_rmse(estimator, x, y):
    # Get predictions
    prediction = estimator.predict_proba(x)
    
    # Calculate the expected value of each document
    result = np.zeros(x.shape[0])
    for i in range(prediction.shape[0]):
        result[i] = expected_value(prediction[i])
    
    # Convert actual y from string to int
    y_int = np.array(list(map(int, y)))
    
    # Calculate and return rmse
    return np.sqrt(np.mean(np.square(result - y_int)))

"""
Calculate the expected value based on the given probabilities.

INPUT:
probability: Array corresponding to probability document has index as label.

OUTPUT:
returns the expected value of the document.
"""
def expected_value(probability):
    return sum(probability * np.arange(1, 6))

def main():
    ############################ Welcome Text 
    totalStart = time.perf_counter()
    
    # Print general info about model
    print_welcome_text(LOAD_MODEL, VERSION, MIN_DF, MAX_DF, MAX_FEATURE, 
                       MIN_GRAM, MAX_GRAM, USE_MINI, MODEL_TYPE, C, SOLVER, 
                       PENALTY, L1_RATIO, TEST_RATIO, MAX_ITER, SAMPLE_STRAT)
    
    
    ############################### VECTORIZING ###############################
    
    print("Begin vectorizing...")
    vectorizeStart = time.perf_counter()
    
    # Get the tf vector and corresponding tf-idf matrix for the training set
    tf, x_train = vectorize_text(TRAIN_CSV, VECTOR, X_TRAIN, LOAD, min_df=MIN_DF, 
                                 max_df = MAX_DF, max_feature=MAX_FEATURE, 
                                 min_n=MIN_GRAM, max_n=MAX_GRAM)

    vectorizeEnd = time.perf_counter()
    elapsed = vectorizeEnd - vectorizeStart
    print("Vectorizing complete!")
    print("Vectorizing took: {}\n".format(convert_elapsed_time(elapsed)))

    # Grab the labels (stars) for the training set
    train_star = pd.read_csv(TRAIN_CSV, usecols=['star'])
    # Use a string to represent the categories
    y_train = list(map(str, train_star['star']))
    
    ############################ Load / Train Model ###########################
    
    if LOAD_MODEL:
        # Load the model
        print("Loading model...")
        logReg = load(open(MODEL_IN, 'rb'))
        print("Model loaded!\n")
    else:
        # Train the model
        print("Begin training...")
        trainingStart = time.perf_counter()
        
        logReg = train(x_train, y_train, MODEL, PENALTY, L1_RATIO, MAX_ITER, C,
                       SOLVER, TEST_RATIO, SAMPLE_STRAT)
        
        trainingEnd = time.perf_counter()
        elapsed = trainingEnd - trainingStart
        print("Training complete!")
        print("Training took: {}\n".format(convert_elapsed_time(elapsed)))
        
    ######################### Predict on Testing Set ##########################
    
    predictStart = time.perf_counter()
    
    # Print the rmse of all training data
    full_score = calculate_rmse(logReg, x_train, y_train)
    print("Rmse of entire training set: {}".format(full_score))
    
    print("Begin prediction of testing set...")
    
    # Predict the values of the test data
    # Saves the test predictions to PREDICTION_CSV
    predict(logReg, tf, TEST_CSV, PREDICTION_CSV)
    
    predictEnd = time.perf_counter()
    print("Time to predict: {}\n".format(convert_elapsed_time(predictEnd - predictStart)))
    
    ########################### Print Model Results ###########################
    
    # Plot a histogram of predicted values and their expected proportions
    visualize(TRAIN_CSV, PREDICTION_CSV)
    
    totalEnd = time.perf_counter()
    elapsed = totalEnd - totalStart
    print("Finished!")
    print("Total time: {}\n".format(convert_elapsed_time(elapsed)))

if __name__ == '__main__':
    main()
