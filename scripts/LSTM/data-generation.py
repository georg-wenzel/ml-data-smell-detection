import random
import pandas
import string
import os

###SCRIPT DESCRIPTION###
# This script provides data generation for the LSTM classifier
###SCRIPT INPUT###
# This script should be provided with two .txt files
# Each file should contain a list of date formats (newline separated),
# followed by an empty line, followed by a list of time formats
# (newline separated).
###SCRIPT OUTPUT###
# This script will generate train.csv, test.csv and validate.csv files
# containing date entries with labeled format identifier and data smells
# according to the formats given in the input files.
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
# MMM and MON are specific keys which represent a name (month or month shorthand) based on month index
mmm_dict = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
    'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
mon_dict = ['January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December']
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
    'z': [("gt", 0), ("lt", 0)],
    # DATEONLY means that the date part is displayed, not the time part
    'additional': ["DATEONLY"]
}
# here you can configure how often each smell should be produced in training and testing 
# (testing multiplier is also used for validation).
incorrect_train_multiplier = [2,2,2,6,6,10,6]
incorrect_test_multiplier = [10,10,10,10,10,10,10]
# these are all the smells we are trying to cover
#   - date(time) as string
#   - date as datetime
#   - shorthand date (ambiguous date)
#   - shorthand datetime (ambiguous date in a datetime format)
#   - shorthand time (ambiguous time)
#   - missing timezone (ambiguous time)
# You may REMOVE entries from this array to exclude certain data smells, but not add or duplicate
# keys without changing the functionality of the script
conditions_incorrect = ["DATEASSTRING", "DATETIMEASSTRING",
    "DATEASDATETIME", "SHORTHANDDATE", "SHORTHANDDATETIME", "SHORTHANDTIME", "MISSINGTIMEZONE"]
###SCRIPT BEGIN####

formats_train = input("Path to training formats file: ")
formats_test = input("Path to test formats file: ")
output_folder = input("Path to output FOLDER: ")

#holds all the training and validation formats
dateformats = []
timeformats = []
dateformats_test = []
timeformats_test = []

# first, load all possible formats from the text files
with open(formats_train) as f:
    formats = [line.strip('\n') for line in f.readlines()]
    # the end of dates and start of times is indicated by a newline
    for i, f in enumerate(formats):
        if not f:
            dateformats = formats[:i]
            timeformats = formats[i+1:]
            break

with open(formats_test) as f:
    formats = [line.strip('\n') for line in f.readlines()]
    # the end of dates and start of times is indicated by a newline
    for i, f in enumerate(formats):
        if not f:
            dateformats_test = formats[:i]
            timeformats_test = formats[i+1:]
            break

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
def create_correct_entries(dateformat, timeformat, date_id, datetime_id):
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

            # if DATEONLY, only create a date without the additional time syntax
            if key == 'additional' and tup == "DATEONLY":
                datestring = dateformat
            else:
                datestring = dateformat + " " + timeformat

            # replace each part of the string
            datestring = datestring.replace("dd", str(d).zfill(2))
            datestring = datestring.replace("yyyy", str(y))
            datestring = datestring.replace("hh", str(h).zfill(2))
            datestring = datestring.replace("mm", str(m).zfill(2))
            datestring = datestring.replace("ss", str(s).zfill(2))
            datestring = datestring.replace("s", str(ms).zfill(4))
            # S is replaced with AM or PM if the hour is > 0 and < 13, otherwise it is omitted since AM or PM does not make sense
            datestring = datestring.replace(
                "S", random.choice(valid['S']) if (h < 13 and h > 0) else "")
            # z is replaced with (+/-)xx:00 where xx is between 0 and 12
            datestring = datestring.replace(
                "z", (("+" + str(z).zfill(2)) if z >= 0 else str(z).zfill(3)) + ":00")
            datestring = datestring.replace("MMM", mmm_dict[M-1])
            datestring = datestring.replace("MM", str(M).zfill(2))
            datestring = datestring.replace("mon", mon_dict[M-1])

            # this will always generate valid dates, which we label as 0
            labels.append([datestring, 0, date_id if key == 'additional' and tup == 'DATEONLY' else datetime_id])

    # return the set of generated labels
    return labels


# for each date and time format, create one instance of each smell
def create_incorrect_entries(dateformat, timeformat, date_id, datetime_id, incorrect_example_multiplier):
    # labels that we will return later
    labels = []

    # iterate over every condition
    for key in conditions_incorrect:
        #create # of examples according to the multiplier
        for _ in range(incorrect_example_multiplier[conditions_incorrect.index(key)]):
            # generate a valid date
            # if DATEASSTRING or SHORTHANDDATE, only create a date without the additional time syntax
            if key == 'DATEASSTRING' or key == "SHORTHANDDATE":
                datestring = dateformat
            else:
                datestring = dateformat + " " + timeformat

            # create random format in constraints
            y = random.randint(valid['y'][0], valid['y'][1])
            M = random.randint(valid['M'][0], valid['M'][1])
            monthdays = (1, valid['d'][1][M-1])
            d = random.randint(monthdays[0], monthdays[1])
            if key != "DATEASDATETIME":
                # make sure the hour is between 1 and 12 if the smell is SHORTHANDTIME
                h = random.randint(1, 12) if key == "SHORTHANDTIME" else random.randint(
                    valid['h'][0], valid['h'][1])
                m = random.randint(valid['m'][0], valid['m'][1])
                s = random.randint(valid['s'][0], valid['s'][1])
                ms = random.randint(valid['ms'][0], valid['ms'][1])
                z = random.randint(valid['z'][0], valid['z'][1])
            else:
                h = random.choice([0,12])
                m = 0
                s = 0
                ms = 0
                z = 0

            # create the datestring
            datestring = datestring.replace("dd", str(d).zfill(2))
            datestring = datestring.replace("yyyy", str(abs(y) % 100).zfill(2)) if (key == "SHORTHANDDATE" or key == "SHORTHANDDATETIME") else datestring.replace("yyyy", str(y).zfill(4))
            datestring = datestring.replace("hh", str(h).zfill(2))
            datestring = datestring.replace("mm", str(m).zfill(2))
            datestring = datestring.replace("ss", str(s).zfill(2))
            datestring = datestring.replace("s", str(ms).zfill(4))
            # S is replaced with AM or PM, unless the smell dictates otherwise
            if key == "SHORTHANDTIME":
                datestring = datestring.replace("S", "")
            elif key == "DATEASDATETIME":
                datestring = datestring.replace("S", "AM" if h == 12 else "")
            else:
                datestring = datestring.replace("S", random.choice(valid['S']) if (h < 13 and h > 0) else "")
            # z is replaced with (+/-)xx:00 where xx is between 0 and 12
            tz_string = (("+" + str(z).zfill(2)) if z >= 0 else str(z).zfill(3)) + ":00"
            datestring=datestring.replace("z",  tz_string if key != "MISSINGTIMEZONE" else "")
            datestring=datestring.replace("mon", mon_dict[M-1])
            datestring = datestring.replace("MMM", mmm_dict[M-1])
            datestring = datestring.replace("MM", str(M).zfill(2))

            if (key == "DATEASSTRING" or key == "DATETIMEASSTRING"):
                datestring = "\"" + datestring + "\""

            labels.append([datestring, conditions_incorrect.index(key)+1, date_id if key == "DATEASSTRING" or key == "SHORTHANDDATE" else datetime_id])
    return labels

#holds the rows of each data frame
labels_train = []
labels_test = []
labels_validate = []

# we want a combination of each time and date pair
for a, dateformat in enumerate(dateformats):
    for b, timeformat in enumerate(timeformats):
        datetime_id = a * (len(timeformats)+1) + b
        date_id = a * (len(timeformats)+1) + len(timeformats)
        labels_train += (create_correct_entries(dateformat, timeformat, date_id, datetime_id))
        labels_train += (create_incorrect_entries(dateformat, timeformat, date_id, datetime_id, incorrect_train_multiplier))
        labels_validate += (create_correct_entries(dateformat, timeformat, date_id, datetime_id))
        labels_validate += (create_incorrect_entries(dateformat, timeformat, date_id, datetime_id, incorrect_test_multiplier))

# for the test set, iterate over the test formats instead
for a, dateformat in enumerate(dateformats_test):
    for b, timeformat in enumerate(timeformats_test):
        datetime_id = a * (len(timeformats_test)+1) + b
        date_id = a * (len(timeformats_test)+1) + len(timeformats_test)
        labels_test += (create_correct_entries(dateformat, timeformat, date_id, datetime_id))
        labels_test += (create_incorrect_entries(dateformat, timeformat, date_id, datetime_id, incorrect_test_multiplier))

df=pandas.DataFrame(labels_train, columns=['Date', 'true_label', 'Format'])
df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
df=pandas.DataFrame(labels_test, columns=['Date', 'true_label', 'Format'])
df.to_csv(os.path.join(output_folder, "test.csv"), index=False)
df=pandas.DataFrame(labels_validate, columns=['Date', 'true_label', 'Format'])
df.to_csv(os.path.join(output_folder, "validate.csv"), index=False)