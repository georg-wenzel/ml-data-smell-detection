import pandas as pd
import numpy as np

###SCRIPT DESCRIPTION###
# This script provides statistical analysis for string distance labeled data. (i.e. Dedupe)
###SCRIPT INPUT###
# The .csv file given to this script should be equivalent to the labeled output generated by the corresponding
# data-generation.py script. i.e. each "true_label" is an integer that is the same for each entry in a real cluster
# each Cluster contains exactly one ground truth "Name" with the "Data Smell" column set to ""
# and exactly one "Name" for each "Data Smell". Each "cluster" corresponds to a cluster detected by Dedupe.
###SCRIPT OUTPUT###
# This script provides the following values in a .csv file
#   - Amount of dataset-wide "Nonsense Clusters" (i.e. clusters that do not correspond to a single real cluster base entry)
#   - Amount of real occurences, incorrect clusters, correct clusters, and undetected clusters (i.e. entries with a unique cluster ID)
#     corresponding to each data smell within the dataset
###SCRIPT CONFIGURATION###
# provides the name of the data smell column in generated datasets
smell_col_name = "Data Smell"
###SCRIPT BEGIN####

input_file = input("Path to results file: ")
output_file = input("Path to output file: ")

data = pd.read_csv(input_file)
# get all unique data smells (dropping NaN, i.e. no data smell)
smells = data[smell_col_name].dropna().unique()
#get all real clusters
cluster_ids = data['true_label'].unique()

real_dedupe_clusters = [-x for x in cluster_ids]
# for each real cluster
for i in cluster_ids:
    #get the cluster ID dedupe has assigned to the non-smelly value
    dedupe_cluster = data[(data['true_label'] == i) & (data[smell_col_name].isnull())].iloc[0]['cluster']
    #replace the dedupe cluster ID with the negation of the real cluster ID
    data['cluster'].replace({dedupe_cluster: -i}, inplace=True)
# at this point, every dedupe cluster is mapped to its corresponding base-entry cluster ID (*-1)

#dictionary which holds the amount of smells we detected properly and improperly detected for each smell
detected_smells = dict()
incorrect_smells = dict()
undetected_smells = dict()
for smell in smells:
    #correctly detected is the # of data smell entries where the dedupe cluster id matches up with the real cluster id
    detected_smells[smell] = data[(data[smell_col_name] == smell) & (data['cluster'] == -data['true_label'])].shape[0]
    #incorrectly detected is the # of data smell entries where the dedupe cluster id does not match up with the real cluster id, but with another real cluster
    incorrect_smells[smell] = data[(data[smell_col_name] == smell) & ~(data['cluster'] == -data['true_label']) & (data['cluster'].isin(real_dedupe_clusters))].shape[0]
    #undetected smells is the # of data smell entries where the dedupe cluster id does not match up with any real cluster, signalling that the smell has not been matched to any cluster
    undetected_smells[smell] = data[(data[smell_col_name] == smell) & ~(data['cluster'] == -data['true_label']) & ~(data['cluster'].isin(real_dedupe_clusters))].shape[0]

#get the counts of entries in each dedupe cluster which is not linked to a real cluster 
dedupe_cluster_counts = data[~(data['cluster'].isin(real_dedupe_clusters))]['cluster'].value_counts()
output = []
#Append the count of nonsense clusters, i.e. clusters with more than 1 entry that do not correspond to any real cluster
output.append(["Nonsense Clusters: " + str(dedupe_cluster_counts[dedupe_cluster_counts > 1].shape[0]),None,None,None,None])
#append the smell-wise number of correctly, incorrectly and not at all matched entries
for smell in smells:
    output.append([smell, len(cluster_ids), incorrect_smells[smell], detected_smells[smell], undetected_smells[smell]])

#output to dataframe
df=pd.DataFrame(output, columns=[smell_col_name, 'Real Occurences', 'Incorrectly Clustered', 'Correctly Clustered', 'Undetected'])
df.to_csv(output_file, index=False)