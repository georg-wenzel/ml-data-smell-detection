import os
from django.db import models
from django.contrib.auth.models import User

### HELPER FUNCTIONS
#helper method for setting the storage path for uploaded datasets
def user_directory_path(instance, filename):
    return 'data/datasets/user_{0}/{1}'.format(instance.user.id, filename)

#helper method for setting the storage path for initialized models
def user_model_directory_path(instance, filename):
    #this path does not use the provided filename due to potential complications when trying to apply a tempfile's filename.
    return 'data/models/user_{0}/model'.format(instance.user.id)

### DATASET MODELS
#Describes a single dataset belonging to a specific user
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True)
    upload = models.FileField(upload_to=user_directory_path)
    col_length = models.IntegerField(null=True)
    row_length = models.IntegerField(null=True)
    #set required=False so form validation succeeds, then set user manually
    user = models.ForeignKey(User, unique=False, on_delete=models.CASCADE, null=True)

    #return the filename from the full filepath (stored under upload)
    def get_filename(self):
        return os.path.basename(self.upload.name)

#describes a single column metadata within a dataset using its name, datatype, and index within the dataset 
class  Column(models.Model):
    name = models.CharField(max_length=255)
    dtype = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True)


### AGENT MODELS
#describes a type of agent (i.e. dedupe, RNN)
class AgentType(models.Model):
    name = models.CharField(max_length=255)

#Describes a single ML agent belonging to a specific user
class Agent(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True)
    agent_type = models.ForeignKey(AgentType, unique=False, on_delete=models.RESTRICT, null=True)
    user = models.ForeignKey(User, unique=False, on_delete=models.CASCADE, null=True)
    #the dataset the model was (or should be) trained on
    dataset = models.ForeignKey(Dataset, unique=False, on_delete=models.RESTRICT, null=True)
    #if another key is used to load an external model (i.e. gensim pretrained models), this column can be used instead of dataset
    external_set_key = models.CharField(max_length=255, null=True)
    #determines if the external set is prepared for validation
    external_prepared = models.BooleanField(null=True, default=False)
    #training iterations this agent has been through (e.g. for Dedupe the number of labels provided)
    iterations = models.IntegerField(default=0)
    #file location where the model info is stored during training
    model = models.FileField(upload_to=user_model_directory_path, null=True)
    #file location where the final model settings to analyze data are stored
    settings = models.FileField(upload_to=user_model_directory_path, null=True)

    def status(self):
        if self.settings or self.external_prepared:
            return "Trained (" + str(self.iterations) + " iterations)"
        elif self.model:
            return "In Training"
        elif self.dataset or self.external_set_key:
            return "Training Data Defined"
        else:
            return "Uninitialized"

    def short_description(self):
        cs = AgentColumn.objects.filter(agent = self)
        description = self.name + " ("
        if self.settings:
            for c in cs:
                description = description + c.name + ", "
            description = description[:-2] + ")"
        else:
            description = description + self.external_set_key + ")"
        return description

#Describes column metadata for a single column that an agent is trained on
class AgentColumn(models.Model):
    name = models.CharField(max_length=255)
    dtype = models.CharField(max_length=255)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE)