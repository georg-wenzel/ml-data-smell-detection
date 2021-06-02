# this script creates labelled datetime strings per the formats defined in date-formats.txt (newline separated)
import random
import pandas
import string
import os

###SCRIPT DESCRIPTION###
# This script provides data generation for the anomaly classifier
###SCRIPT INPUT###
# This script needs to be provided with one .txt file containing two lines
# The first line is the format in the training file, the second line is
# The format which makes up 50% of the test file.
###SCRIPT OUTPUT###
# This script will generate train.csv and test.csv files
# The train.csv file will contain a single column with dates in the given format
# The test.csv file will contain a mix of the two dates given in the input
# As well as an additional column labelling the formats as 0 or 1
###SCRIPT CONFIGURATION###
# this defines the general valid range for each format placeholder
valid = {
    'y': (1970, 2021),
    'M': (1, 12),
    # valid days are dependent on month - leap years are not included here
    'd': (1, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
    'h': (0, 23),
    'm': (0, 59),
    's': (0, 59),
    'ms': (0, 999),
    'S': ["AM", "PM"],
    'z': (-12, 12)
}
# this defines a specific condition for each correct date
# i.e. the first date must have a valid year between min year and 2000
# this is only used to make sure the random date range generated sufficiently covers most of the valid date range for each format
conditions_correct = {
    'y': [("lt", 2000), ("gt", 2000)],
    'M': [("eq", 1), ("eq", 2), ("eq", 3), ("eq", 4), ("eq", 5), ("eq", 6), ("eq", 7), ("eq", 8), ("eq", 9), ("eq", 10), ("eq", 11), ("eq", 12)],
    'd': [("lt", 10)],
    'h': [("lt", 10), ("gt", 12)],
    'm': [("lt", 10)],
    's': [("lt", 10)],
    'ms': [("lt", 10), ("btw", 10, 100)],
    'z': [("gt", 0), ("lt", 0)]
}
# Defines the number of entries generated per format
# number is multiplied with the number of correct_conditions (e.g. 23 correct conditions * 100 iterations = 2300 entries per format)
num_iterations = 100
###SCRIPT BEGIN###

input_file = input("Path to formats file: ")
output_folder = input("Path to output FOLDER: ")

# a helper method which simplifies applying the conditions in the above array
def tuple_constraint(base, tuple):
    if tuple[0] == "eq":
        return tuple[1]
    if tuple[0] == "lt":
        return random.randint(base[0], tuple[1]-1)
    if tuple[0] == "gt":
        return random.randint(tuple[1], base[1])
    if tuple[0] == "btw":
        return random.randint(tuple[1], tuple[2])

# for a given time and date format, create examples which fulfill all conditions (correct)
def create_correct_entries(dateformat, label=-1):
    # labels returned in the end (array of date string and label)
    labels = []

    # iterate over every condition
    for key in conditions_correct:
        for tup in conditions_correct[key]:
            # create random format in constraints
            y = tuple_constraint(valid['y'], tup) if key == 'y' else random.randint(
                valid['y'][0], valid['y'][1])
            M = tuple_constraint(valid['M'], tup) if key == 'M' else random.randint(
                valid['M'][0], valid['M'][1])
            monthdays = (1, valid['d'][1][M-1])
            d = tuple_constraint(monthdays, tup) if key == 'd' else random.randint(
                monthdays[0], monthdays[1])
            h = tuple_constraint(valid['h'], tup) if key == 'h' else random.randint(
                valid['h'][0], valid['h'][1])
            m = tuple_constraint(valid['m'], tup) if key == 'm' else random.randint(
                valid['m'][0], valid['m'][1])
            s = tuple_constraint(valid['s'], tup) if key == 's' else random.randint(
                valid['s'][0], valid['s'][1])
            ms = tuple_constraint(valid['ms'], tup) if key == 'ms' else random.randint(
                valid['ms'][0], valid['ms'][1])
            z = tuple_constraint(valid['z'], tup) if key == 'z' else random.randint(
                valid['z'][0], valid['z'][1])

            datestring = dateformat

            # replace each part of the string
            datestring = datestring.replace("dd", str(d).zfill(2))
            datestring = datestring.replace("yyyy", str(y))
            datestring = datestring.replace("hh", str(h).zfill(2))
            datestring = datestring.replace("mm", str(m).zfill(2))
            datestring = datestring.replace("ss", str(s).zfill(2))
            datestring = datestring.replace("s", str(ms).zfill(4))
            # S is replaced with AM or PM if the hour is < 13, otherwise it is omitted since AM or PM does not make sense
            datestring = datestring.replace(
                "S", random.choice(valid['S']) if h < 13 else "")
            # z is replaced with (+/-)xx:00 where xx is between 0 and 12
            datestring = datestring.replace(
                "z", (("+" + str(z).zfill(2)) if z >= 0 else str(z).zfill(3)) + ":00")
            datestring = datestring.replace("MM", str(M).zfill(2))

            if label != -1: labels.append([datestring, label])
            else: labels.append([datestring])

    # return the set of generated labels
    return labels

labels_train = []
labels_test = []

with open(input_file) as f:
    formats = [line.strip('\n') for line in f.readlines()]
    # the end of dates and start of times is indicated by a newline
    for i in range(100):
        labels_train += create_correct_entries(formats[0])
    for i in range(100):
        labels_test += create_correct_entries(formats[0], 0)
        labels_test += create_correct_entries(formats[1], 1)
    
df=pandas.DataFrame(labels_train, columns=['Date'])
df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
df=pandas.DataFrame(labels_test, columns=['Date', 'true_label'])
df.to_csv(os.path.join(output_folder, "test.csv"), index=False)
