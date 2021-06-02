import os
import csv
from unidecode import unidecode
import tempfile

# Utility functions for Dedupe

#Sets column to None if it represents an empty string
#https://dedupeio.github.io/dedupe-examples/docs/csv_example.html
def preprocess(column):
    column = unidecode(column)

    if not column:
        column = None
    return column

#load data from csv file into in-memory dictionary
#https://dedupeio.github.io/dedupe-examples/docs/csv_example.html
def read_data(file):
    data_d = {}
    with open(os.path.normpath(file)) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            #preprocessing here if necessary
            clean_row = [(k, preprocess(v)) for (k, v) in row.items()]
            row_id = i
            data_d[row_id] = dict(clean_row)
    return data_d

#setup fields for dedupe from a list of AgentColumn model
def set_fields(columns):
    fields = []
    for c in columns:
        fields.append({'field': c.name, 'type': c.dtype})
    return fields

#trains a dedupe model and returns file paths to the model and settings file
def train_model(model):
    # train model
    model.train()

    # write training and settings to files and return them
    mdata = tempfile.NamedTemporaryFile('w', delete=False)
    model.write_training(mdata)
    mdata.close()

    settings = tempfile.NamedTemporaryFile('wb', delete=False)
    model.write_settings(settings)
    settings.close()

    #frees up memory
    model.cleanup_training()

    return mdata.name, settings.name