"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 12/3/20
Logistic Regression Classifier for Yelp Dataset

Creates and saves to file predictions for the given dataset along with each 
document's id.
"""

import numpy as np
import pandas as pd

"""
Calculates and saves to a file the predictions of the testing set.

INPUT:
estimator: The model that the dataset is to be predicted with.
tf: The tf vectorizer that will transform the text into a tf-idf matrix.
input_csv: Filepath to the csv file to predict the labels of.
output_csv: Filepath to save the predictions to.

OUTPUT:
Saves the predictions to output_csv.
"""
def predict(estimator, tf, input_csv, output_csv):
    # Read in testing set
    test_set = pd.read_csv(input_csv)
    
    # Get text and id of all entries
    id_test = list(map(str, test_set['Id']))
    test_text = test_set['text'].tolist()
    
    # Transform text into tf-idf matrix
    x_test = tf.transform(test_text)
    
    print("Successfully loaded testing set, with shape {}.".format(test_set.shape))

    # Get matrix containing probabilities of each label, by entry
    predictions = estimator.predict_proba(x_test)
    
    # Calculate the expected value of each document
    result = np.zeros(test_set.shape[0])
    for i in range(predictions.shape[0]):
        result[i] = expected_value(predictions[i])

    # Combine ID with prediction and write to a file
    with open(output_csv, 'w') as output:
        output.write('"Id","Expected"\n')
        for i in range(len(result)):
            output.write("{},{}\n".format(id_test[i], result[i]))
    
    print("Predictions for the testing data have been written.\n")
    
"""
Calculate the expected value based on the given probabilities.

INPUT:
probability: Array corresponding to the probability document has (index) as label.

OUTPUT:
returns the expected value of the document.
"""
def expected_value(probability):
    return sum(probability * np.arange(1, 6))