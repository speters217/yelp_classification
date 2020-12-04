"""
Author: Samuel Peters (sjpeters3@wisc.edu)
Date: 12/3/20
Logistic Regression Classifier for Yelp Dataset

Either loads or creates and saves a fitted vectorizer. The vectorizer is returned 
along with the tf-idf matrix of the training set.
"""

import pandas as pd
from pickle import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


"""
If load is true, then the fitted vectorizer is loaded and fit to the training 
set. Both are then returned.

Otherwise, loads in the training set, fits a vectorizer to it, and saves 
the vectorizer and then returns it and the tf-idf matrix of the training set.

INPUT:
training_csv: Filepath where the training set csv file is located.
vector: Filepath to store the fitted vectorizer at.
load: Whether or not to load a pre-made vectorizer.
min_df: Minimum document frequency.
max_df: Maximum document frequency.
max_feature: Vocabulary will only have the top max_features ordered by
    term frequency across the corpus. Deafsult value is None.
min_n: Min n-gram to use. Default is 1.
max_n: Max n-gram to use. Default is 1.
    
OUTPUT:
tf: The idf vector.
x_train: The slice of matrix that corresponds to the training set.
"""
def vectorize_text(training_csv, vector, x_train_matrix, load, min_df=1, max_df=1, 
                   max_feature=None, min_n=1, max_n=1):
    # If there is a saved tf file, we load and use it.
    if load:            
        print("Loading pre-existing data...")
        tf, x_train = load_matrix(vector, x_train_matrix, min_df, max_df, 
                                  max_feature, min_n, max_n)
    # Otherwise we must calculate everything.
    else:
        print("Building vectors from scratch...")
        tf, x_train = build_matrix(training_csv, vector, x_train_matrix, min_df, 
                                   max_df, max_feature, min_n, max_n)
        
    print("Successfully loaded the tf-idf matrix, with shape {}.".format(
        x_train.shape))
        
    return tf, x_train

"""
Helper function for vectorize_text. See above.
"""
def load_matrix(vector, x_train_matrix, min_df, max_df, 
                max_feature, min_n, max_n):
    
    # Load stored data
    tf = load(open(vector, 'rb'))
    x_train = load(open(x_train_matrix, 'rb'))
    
    return tf, x_train

"""
Helper function for vectorize_text. See above.
"""
def build_matrix(training_csv, vector, x_train_matrix, min_df, max_df, 
                 max_feature, min_n, max_n):
    # Read in training set with only the text column
    train_csv = pd.read_csv(training_csv, usecols=['text'])
    
    # Get the text field in list format
    train_text = train_csv['text'].tolist()
    
    # Initialize TFID arguments
    # Note that when called, text will be cleaned by tokenize
    tf = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_feature,
                         strip_accents='ascii', analyzer='word',
                         tokenizer=tokenize, ngram_range=(min_n, max_n))
    
    # Vectorize the training set using the tf-idf of the training set
    x_train = tf.fit_transform(train_text)
    
    # Store the tf_vectorizer and tf-idf matrix
    dump(tf, open(vector, 'wb'))
    dump(x_train, open(x_train_matrix, 'wb'))
    
    return tf, x_train

"""
Tokenizes the given text with nltk's tokenizer. Text is made into lowercase,
stop words are filtered out, and all tokens are stemmed using nltk's snowball.

INPUT:
text: A block of text to be tokenized and returned.

OUTPUT:
returns a tokenized list of the text.
"""
def tokenize(text):
    # Stopwords to be removed
    stop = set(stopwords.words("english"))
    
    # Tokenize the text, filter out stop words, then make lowercase.
    trimmed_tokens = [w.lower() for w in word_tokenize(text) if w not in stop]
    
    # Snowball stemmer to stem words
    stemmer = SnowballStemmer("english")
    
    # Stem the tokens using snowball
    stemmed_tokens = [stemmer.stem(t) for t in trimmed_tokens]

    return stemmed_tokens