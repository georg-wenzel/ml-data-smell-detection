import os
import pandas
import gensim.utils
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk

# Utility functions for gensim

#load data from a specific column of a csv file into in-memory list (documents) of lists (word tokens)
def get_corpus(file, column):
    csv = pandas.read_csv(os.path.normpath(file))
    sentences = csv[column].to_list()
    tokenized = list(map(lambda x: gensim.utils.simple_preprocess(x, min_len=2, max_len=15), sentences))
    return tokenized

#load data from a specific column of a csv file into a list of words with an occurence of min 5 times, excluding stopwords
#these word vectors can be used to check for vocabulary using the key_to_index (list vocab) and has_index_for (check for word in vocab) methods
def get_vocabulary(file, column):
    nltk.download('stopwords')

    csv = pandas.read_csv(os.path.normpath(file))
    sentences = csv[column].to_list()
    tokenized = list(map(lambda x: gensim.utils.simple_preprocess(x, min_len=2), sentences))
    #build vocabulary without training
    w2v = Word2Vec()
    w2v.build_vocab(corpus_iterable=tokenized)
    words = w2v.wv.key_to_index
    #remove stopwords
    return list(set(words) - set(stopwords.words('english')))


#load data from a specific column of a csv file into in-memory list of sentences
#this can be used to check against all rows in a specific column of a dataset.
def get_unfiltered(file, column):
    csv = pandas.read_csv(os.path.normpath(file))
    sentences = csv[column].to_list()
    return sentences