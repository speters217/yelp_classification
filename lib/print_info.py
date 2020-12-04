"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 12/3/20
Logistic Regression Classifier for Yelp Dataset

This file prints some welcome text and general information about the model.
"""

import numpy as np
from datetime import datetime

"""
Prints some text describing the parameters of the current model.
"""
def print_welcome_text(load_model, version, min_df, max_df, max_features, 
                       min_gram, max_gram, use_mini, model_type, c, solver, 
                       penalty, l1_ratio, test_ratio, max_iter, sample_strat):
    
    print("Using a {} model".format(model_type))
    if load_model:
        print("Model is being loaded\n\n")
    else:
        print("Version: {}".format(version))
        
        time_started = datetime.now()
        start_time = "{}/{}/{} {}:{}:{}".format(time_started.month, 
                                                time_started.day, 
                                                time_started.year, 
                                                time_started.hour, 
                                                time_started.minute, 
                                                time_started.second)
        print("Start time: {}".format(start_time))
        
        # Handle if percentages are being used
        if min_df < 1:
            min_df = "{:.2f}%".format(min_df)
        if max_df < 1:
            max_df = "{:.2f}%".format(max_df)
    
        print("Vocab cutoff: {}-{}".format(min_df, max_df))
    
        if max_features != None:
            print("Only considering the top {} features ordered by term frequency across the corpus.".format(max_features))
    
        if min_gram == max_gram:
            print("Using only {}-grams".format(min_gram))
        else:
            print("Using {}-grams through {}-grams".format(min_gram, max_gram))
    
        if use_mini:
            print("Using the truncated training set")
        else:
            print("Using the full training set")
            
        print("Solver: {}".format(solver))
        
        print("Inverse regularization strength: {}".format(c))
        
        print("Penalty: {}".format(penalty))
        if penalty == 'elasticnet':
            print("Elastic-Net mixing: {}".format(l1_ratio))

        print("Max # of iterations to converge: {}".format(max_iter))
        
        print("SMOTE strategy: {}".format(sample_strat))
        
        print("Testing set ratio: {}\n\n".format(test_ratio))
        
        
"""
Converts float seconds into an hours:minutes:seconds string.
"""
def convert_elapsed_time(elapsed):
    seconds = np.floor(elapsed % 60)
    minutes = np.floor(elapsed / 60)
    hours = np.floor(minutes / 60)
    minutes = minutes % 60
    return "{}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))