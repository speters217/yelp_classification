"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 12/3/20
Logistic Regression Classifier for Yelp Dataset

This file cleans the given yelp datasets by removing unwanted columns and 
performing minimal text cleaning. The cleaned versions as well as a mini version 
of the training set are saved.
"""

import os
import pandas as pd
import numpy as np
import re

DIRECTORY = os.getcwd() # Current Directory
DATASET_DIRECTORY = 'Datasets' # Directory that holds all datasets

# Filepaths for the input datasets
TRAIN_IN = os.path.join(DIRECTORY, DATASET_DIRECTORY,'yelp_train_in.csv')
TEST_IN = os.path.join(DIRECTORY, DATASET_DIRECTORY, 'yelp_test_in.csv')

# Filepaths for the output datasets
TRAIN_CSV = os.path.join(DIRECTORY, DATASET_DIRECTORY, 'yelp_train.csv')
TEST_CSV = os.path.join(DIRECTORY, DATASET_DIRECTORY, 'yelp_test.csv')
TRAIN_MINI = os.path.join(DIRECTORY, DATASET_DIRECTORY, 'yelp_train_mini.csv')

# Thanks to Greenstick at https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

"""
Uses regular expressions to clean the text, then returns it all in lowercase.
"""
def clean_text(text):
    # Make punctuation its own word by adding spaces before and after it
    new_text = re.sub(r"""([.,;!?()$"\\/#%@&*\{\}\[\]:~`])""", r' \1 ', text)

    # Add a space before all apostrophes to make it a word with what's after it
    new_text = re.sub('\'', ' \'', new_text)

    # Consolidate multiple spaces into one
    new_text = re.sub('\s{2,}', ' ', new_text)

    # Remove newlines
    new_text = re.sub('\n', ' ', new_text)

    # Convert all letters to lower case and return
    return new_text.lower()

"""
Loads in the datasets, strips the unwanted columns, cleans the text, and
saves the output datasets to file.
"""
def parse():
    # Only load in the wanted columns of the datasets
    old_train = pd.read_csv(TRAIN_IN, usecols=['star', 'text'])
    old_test = pd.read_csv(TEST_IN, usecols=['Id', 'text'])
    
    # Progress bar for parsing text
    progress = 0
    size = old_test.shape[0] + old_train.shape[0]
    printProgressBar(progress, size, prefix = 'Processing Data:', suffix = 'Complete', length = 50)

    # Clean all text in training set
    for row in old_train.iterrows():
        row[1][1] = clean_text(row[1][1])
        progress += 1
        printProgressBar(progress, size, prefix = 'Processing Data:', suffix = 'Complete', length = 50)
        
    # Clean all text in testing set
    for row in old_test.iterrows():
        row[1][1] = clean_text(row[1][1])
        progress += 1
        printProgressBar(progress, size, prefix = 'Processing Data:', suffix = 'Complete', length = 50)
    
    
    # We create a shortened version of the training set to use to quickly test code / parameters
    # Mini train gets first 40% of the full dataset
    train_mini = old_train[0:int(np.floor(old_train.shape[0]*0.4))]    
    
    
    progress = 0
    size = 3
    printProgressBar(progress, size, prefix = 'Writing to files:', suffix = 'Complete', length = 50)
    
    # Write all files
    old_test.to_csv(TEST_CSV, index=False)
    progress += 1
    printProgressBar(progress, size, prefix = 'Writing to files:', suffix = 'Complete', length = 50)

    old_train.to_csv(TRAIN_CSV, index=False)
    progress += 1
    printProgressBar(progress, size, prefix = 'Writing to files:', suffix = 'Complete', length = 50)

    train_mini.to_csv(TRAIN_MINI, index=False)
    progress += 1
    printProgressBar(progress, size, prefix = 'Writing to files:', suffix = 'Complete', length = 50)
    

    
def main():
    parse()

if __name__ == '__main__':
    main()