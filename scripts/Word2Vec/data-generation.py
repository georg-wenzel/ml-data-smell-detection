import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

###SCRIPT DESCRIPTION###
# This script provides a list of synonymous pairs detected by WordNet based on a
# textual column.
###SCRIPT INPUT###
# This script is meant to be used with the Covid-19 NLP text classification dataset
# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
# However, by adjusting the column text is read from, it can be used on any dataset. 
###SCRIPT OUTPUT###
# This script generates a .csv file where the first column is a word, and the
# second is a synonymous word. 
###SCRIPT CONFIGURATION###
# name of the column to extract text from
text_column = "OriginalTweet"
###SCRIPT BEGIN####

input_file = input("Path to input file: ")
output_file = input("Path to output file: ")

data = pd.read_csv(input_file)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#create set of all eligible words
words = []
for sentence in data[text_column]:
    words += [x for x in nltk.word_tokenize(sentence) if len(x) > 1]
wordset = set(words) - set(nltk.corpus.stopwords.words('english'))

#length of word set
print(len(wordset))

stemmer = PorterStemmer()
pairs = []
for i, word in enumerate(wordset):
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            #make sure synonymous word is also occuring in the wordset
            if l.name() in wordset:
                #make sure porter stemmer does not match up (avoid inflections)
                if stemmer.stem(l.name().lower()) != stemmer.stem(word.lower()):
                    #make sure the entry is not already in the list of pairs
                    if not (l.name().lower(), word.lower()) in pairs:
                        if not (word.lower(), l.name().lower()) in pairs:
                            #add to list of pairs
                            pairs.append((word.lower(), l.name().lower()))
    #print every 1000 entries to visualize progress slightly
    if i % 1000 == 0:
        print(i)

#output to file
df=pd.DataFrame([[a,b] for (a,b) in pairs], columns=['Word', 'Synonym'])
df.to_csv(output_file, index=False)