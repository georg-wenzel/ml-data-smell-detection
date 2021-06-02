import requests
import random
import string
import pandas
import os

###SCRIPT DESCRIPTION###
# This script provides data generation for the logistic regression classifier
###SCRIPT OUTPUT###
# this script creates random names which may have an instance of:
# - Truncating Smell (By adding a title, or abbreviating the last name)
# - Casing Smell (By lowercasing the first or last name, or both)
# - Spacing Smell (By removing the space between names)
# The script will generate a names_train.csv, names_test.csv, as well as 
# smell_train.csv and smell_test.csv for each of the 3 smells.
# The main files contain all data smell instances, the smell_xx.csv files
# only contain the corresponding data smell (and pure entries)
###SCRIPT CONFIGURATION###
# Titles used for truncating
prefix_titles = ["Dr. ", "Doctor ", "MSc. ", "BSc. "]
suffix_titles = [", Dr.", ", MSc.", ", BSc."]
# amount of unique names generated (for training and testing each)
# setting this number > 2500 will result in an API error as the randomuser.me
# api only allows up to 5000 names to be fetched
n = 2000
# provides the name of the data smell column in generated datasets
smell_col_name = "Data Smell"
###SCRIPT BEGIN####

#determine output folder
output_folder = input("Path to output FOLDER: ")

#make get request for n people from randomuser.me, 
people_json = requests.get('https://randomuser.me/api/?inc=name&nat=gb&results=' + str(2*n)).json()

#columns
names = []
smell = []
cluster = []

#amount of names excluded due to duplication (only used for output)
excluded_count = 0

#final set of names that will be used
true_names = set()

#generate n names
for i, person in enumerate(people_json["results"]):
    firstname = person["name"]["first"].capitalize()
    lastname = person["name"]["last"].capitalize()

    # skip names that are complete duplicates of one another
    if (firstname + " " + lastname) in true_names:
        excluded_count += 1
        continue
    true_names.add(firstname + " " + lastname)

    #baseline version
    names.append(firstname + " " + lastname)
    smell.append("")
    #truncating smell version
    r = random.random()
    if r < 0.5:
        names.append(random.choice(prefix_titles) + firstname + " " + lastname)
    else:
        names.append(firstname + " " + lastname + random.choice(suffix_titles))
    smell.append("SMELL_TRUNCATING")
    #casing smell version
    r = random.random()
    if r < 0.25:
        names.append(firstname.lower() + " " + lastname)
    elif r < 0.5:
        names.append(firstname + " " + lastname.lower())
    elif r < 0.75:
        names.append((firstname + " " + lastname).lower())
    else:
        name = (''.join(random.choice((str.upper, str.lower))(c) for c in (firstname + " " + lastname)))
        # prevents a rare chase where every character randomly matches the original casing,
        # by making sure the last character is uppercased
        name = name[:len(name) - 1] + str.upper(lastname[len(lastname) - 1])
        names.append(name)

    smell.append("SMELL_CASING")
    #spacing smell version
    r = random.random()
    r_len = random.choice([x for x in range(0,3) if x != (1 if r < 0.33 else 0)])
    if r < 0.33:
        names.append(firstname + r_len * ' ' + lastname)
    elif r > 0.66:
        names.append(r_len * ' ' + firstname + " " + lastname)
    else:
        names.append(firstname + " " + lastname + r_len * ' ')
    smell.append("SMELL_SPACING")
    
    cluster += [i,i,i,i]
    
# we do not need to make adjustments based on the excluded count
# the training set will just get slightly smaller
print("Excluded names:", excluded_count)

df = pandas.DataFrame({'Name': names, smell_col_name: smell, 'true_label': cluster})
#create different subsets of the data
#the first half is the training set, consisting of 4*n entries (no smell, casing, spacing, truncating)
training = df.iloc[:4*n]
training.to_csv(os.path.join(output_folder, "names_train.csv"), index=False)
training_spacing = training[(training[smell_col_name] == "SMELL_SPACING") | (training[smell_col_name] == "")]
training_spacing.to_csv(os.path.join(output_folder, "spacing_train.csv"), index=False)
training_casing = training[(training[smell_col_name] == "SMELL_CASING") | (training[smell_col_name] == "")]
training_casing.to_csv(os.path.join(output_folder, "casing_train.csv"), index=False)
training_truncating = training[(training[smell_col_name] == "SMELL_TRUNCATING") | (training[smell_col_name] == "")]
training_truncating.to_csv(os.path.join(output_folder, "truncating_train.csv"), index=False)
#first half is the testing set, consisting of the remaining entries
testing = df.iloc[4*n:, ]
testing.to_csv(os.path.join(output_folder, "names_test.csv"), index=False)
testing_spacing = testing[(testing[smell_col_name] == "SMELL_SPACING") | (testing[smell_col_name] == "")]
testing_spacing.to_csv(os.path.join(output_folder, "spacing_test.csv"), index=False)
testing_casing = testing[(testing[smell_col_name] == "SMELL_CASING") | (testing[smell_col_name] == "")]
testing_casing.to_csv(os.path.join(output_folder, "casing_test.csv"), index=False)
testing_truncating = testing[(testing[smell_col_name] == "SMELL_TRUNCATING") | (testing[smell_col_name] == "")]
testing_truncating.to_csv(os.path.join(output_folder, "truncating_test.csv"), index=False)