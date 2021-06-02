# Utility functions for Strings (i.e. storing common strings once)

#define common strings
ERR_MISSING_KEY = "The field(s) {0} must be filled in this form."
ERR_INVALID_KEY = "The field '{0}' contains an invalid value."
ERR_UNAUTHORIZED = "The logged in user does not have access to this value: {0}"
MSG_FINISHED_TRAINING = "Your agent {0} has finished training and can now be used."

#define error string for (multiple) column mismatch
#pass tuple of mismatched columns as defined by AgentUtility.dataset_all_columns_match
def ERR_COLUMN_MISMATCH(columns_mismatched):
    err = ("Column type mismatch: " +
            " ".join(list(map(lambda x: "\"" + x[0] + "\": Expected " + str(x[1]) + ", but was " + str(x[2]) + ".", columns_mismatched))))
    return err