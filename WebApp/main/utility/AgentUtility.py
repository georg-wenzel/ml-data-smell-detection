from main.agent_handlers.DedupeHandler import DedupeHandler
from main.agent_handlers.GensimHandler import GensimHandler
from main.agent_handlers.LSTMHandler import LSTMHandler
from main.agent_handlers.AnomalyHandler import AnomalyHandler
import pandas as pd
import numpy as np

# array which defines which handler class is called for which agent type (id = array index + 1)
agent_handlers = [DedupeHandler, GensimHandler, LSTMHandler, AnomalyHandler]

# pass a list of tuples of type [("columnName": np.expected_dtype)]
# function returns an array of all type mismatches in form [("columnName", np.expected, np.actual)]
# this version of the function first opens the dataset
def dataset_all_columns_match_unopened(dataset_path, columns_expected):
    data = pd.read_csv(dataset_path)
    return dataset_all_columns_match(data, columns_expected)

# pass a list of tuples of type [("columnName": np.expected_dtype)]
# function returns an array of all type mismatches in form [("columnName", np.expected, np.actual)]
# this version of the function first assumes an in-memory pandas dataframe as the first argument
def dataset_all_columns_match(dataset, columns_expected):
    mismatches = []

    for key, value in columns_expected:
        if dataset.dtypes[key] != value:
            #clears up "object" datatype
            expected = value if value != np.object else "string"
            true = dataset.dtypes[key] if dataset.dtypes[key] != np.object else "string"

            mismatches.append((key, expected, true))
    return mismatches