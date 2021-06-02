# Utility functions for Django

#simple utility method which checks if all keys are present in a given dictionary
def dict_contains_all(dict, keys):
    return all (k in dict for k in keys)