import nltk
from nltk.corpus import wordnet
import pandas as pd
import numpy as np

###SCRIPT DESCRIPTION###
# This script provides statistical analysis for extracted synonyms
###SCRIPT INPUT###
# This script is passed two .csv files, one being the result of data-generation.py on the target dataset,
# the other being the output of the labeling within the webapp.
###SCRIPT OUTPUT###
# This script provides the following values in a .csv file
# The original synonyms input, with an added column flagging each synonym pair to either be present in the WordNet
# input file, or not. Further analysis is done using manual labeling, as indicated in the thesis.
###SCRIPT BEGIN####

input_file_wn = input("Path to WordNet pairs file: ")
input_file_synonyms = input("Path to detected pairs file: ")
output_file = input("Path to output file: ")

wordnet = pd.read_csv(input_file_wn)
synonyms = pd.read_csv(input_file_synonyms)

wn_synonyms = set(zip(wordnet['Word'], wordnet['Synonym']))

syns = []

#add column whether or not the synonym is in wordnet
for a,b in list(zip(synonyms.word, synonyms.synonym)):
    syns.append([a,b,(a,b) in wn_synonyms or (b,a) in wn_synonyms])

df=pd.DataFrame(syns, columns=['Word', 'Synonym', 'is_wordnet'])
df.to_csv(output_file, index=False)
