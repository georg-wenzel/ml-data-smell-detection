from django.shortcuts import render, redirect
from main.models import Dataset, Agent, AgentColumn, Column
import tempfile
import numpy as np
from main.utility import TensorflowUtility
from main.utility import AgentUtility
from django.core.cache import cache
import os
import tensorflow as tf
from main.utility.DatabaseUtility import safe_get
from main.utility.DjangoUtility import dict_contains_all
from main.utility import StringUtility
from django.core.files.base import File
from django.contrib import messages
from django.core.cache import cache
import pickle

# contains logic for setting up, training and analyzing data using a TensorFlow agent (Anomaly Detection)

class AnomalyHandler:

    # Called when user clicks edit agent the first time
    @staticmethod
    def setup(request, agent):

        # if not a POST request, open the corresponding setup page (selecting a dataset and column to train on)
        if request.method != "POST":
            datasets = Dataset.objects.filter(user=request.user)
            return render(request, 'agents/setup_anomaly.html', {"username": request.user.username, "agent": agent, "datasets": datasets})
        
        #otherwise get dataset and target column
        if not dict_contains_all(request.POST, ["dataset", "columnId"]):
            messages.add_message(request, messages.ERROR,
                        "Not all form data was provided.")
            return redirect('/agents')

        dataset = safe_get(Dataset, id=int(request.POST["dataset"]))
        column = safe_get(Column, id=int(request.POST["columnId"]))
        if(not dataset or not column):
            messages.add_message(request, messages.ERROR,
                    "Form data provided was not valid.")
            return redirect('/agents')

        # make sure columns is part of the dataset & save this dataset and column
        if column.dataset == dataset:
            # make sure column is of type string
            mismatches = AgentUtility.dataset_all_columns_match_unopened(dataset.upload.name, 
                            [(column.name, np.object)])
            if len(mismatches) > 0:
                messages.add_message(
                    request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
                return redirect('/agents')

            agent.dataset = dataset
            agent.save()
            # store column as String
            ac = AgentColumn(name=column.name, dtype="String", agent=agent)
            ac.save()
        else:
            messages.add_message(request, messages.ERROR,
                "Provided columns are not part of the correct dataset.")
            return redirect('/agents')

        return redirect('/agents/train/' + str(agent.id))

    # called when user clicks edit agent on subsequent times (after setting up)
    @staticmethod
    def train(request, agent):
        # check to see if this agent is already in the session
        if not request.session.get('agent_id', False) == agent.id:
            # if not, set it to this agents id
            request.session["agent_id"] = agent.id
            
            # get dataset and column
            dataset = safe_get(Dataset, id=agent.dataset.id)
            column = safe_get(AgentColumn, agent=agent)

            if not dataset or not column:            
                messages.add_message(request, messages.ERROR,
                    "Failed to load column for this dataset.")
                return redirect('/agent')

            # get column from the dataset
            csv = TensorflowUtility.get_csv(dataset.upload.path)
            x = csv[column.name]
            # encode column (equivalent to LSTM), store the length
            encoder = TensorflowUtility.get_encoder(x)
            enc_x, maxlength = TensorflowUtility.encode_x_fixed_length(encoder, x)
            # get class weights and train model
            model = TensorflowUtility.train_anomaly_model(enc_x, maxlength)
            # get maximum reconstruction error of the train data
            preds = model.predict(enc_x)
            msemax = np.percentile(np.mean(np.power(enc_x - preds, 2), axis=1), 99, axis=0)
            # store model in model, encoder, maxlength and maximum reconstruction error in settings
            temppath = os.path.join(tempfile.gettempdir(), next(
                tempfile._get_candidate_names()) + ".keras")
            model.save(temppath)
            temp = open(temppath, 'rb')
            agent.model.save("", File(temp), save=True)
            temp.close()
            #use pickle to serialize the encoder and other data so we can in the future use the model
            f = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            pickle.dump({"encoder": encoder, "threshold": msemax, "input_dim": maxlength}, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            temp = open(f.name, 'rb')
            agent.settings.save("", File(temp), save=True)
            temp.close()

            # save agent iterations
            agent.iterations = agent.iterations + 1
            agent.save()

        messages.add_message(
            request, messages.SUCCESS, "Your agent has been trained successfully.")
        return redirect('/agents')

    # called when user uses this agent on the analyze page
    @staticmethod
    def analyze(request, agent, dataset, **kwargs):
        # get data
        data = TensorflowUtility.get_csv(dataset.upload.path)
        # make sure column is of type string
        mismatches = AgentUtility.dataset_all_columns_match(data, 
                        [(kwargs['column'].name, np.object)])
        if len(mismatches) > 0:
            messages.add_message(
                request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
            return redirect('/analyze')
        # load model from model file
        model = tf.keras.models.load_model(agent.model.path)
        # load encoder and other settings from settings file (deserialize)
        with open(agent.settings.path, mode='rb') as f:
            settings = pickle.load(f)
        encoder = settings["encoder"]
        threshold = settings["threshold"]
        input_dim = settings["input_dim"]
        x = data[kwargs['column'].name]
        # encode data
        enc_x, _ = TensorflowUtility.encode_x_fixed_length(encoder, x, input_dim)
        # classify data
        preds = model.predict(enc_x)
        mse = np.mean(np.power(enc_x - preds, 2), axis=1)
        data['mse'] = mse
        data['class'] = data['mse'] > threshold
        data.sort_values('mse', ascending=True, inplace=True)

        cache.set(request.user.username + "_dataset", data)

        #separate into class 0 and class 1 data
        class_0 = data[data['class'] == 0]
        class_1 = data[data['class'] == 1]

        #extract tuples of sorted column and MSE for display
        class_0_values = list(zip(class_0[kwargs['column'].name], round(class_0.mse, 4)))
        class_1_values =  list(zip(class_1[kwargs['column'].name], round(class_1.mse, 4)))
        
        #extract true counts
        counts = [class_0.shape[0], class_1.shape[0]]

        #extract avg. classwise MSE and MSE threshold
        mse_values = [round(np.mean(class_0['mse']),4), round(np.mean(class_1['mse']),4), round(threshold,4)]

        #display first 5 entries of class 0 and last 5 entries of class 1
        if len(class_0_values) > 5 and len(class_1_values) > 5: example_values = [class_0_values[:5], class_1_values[-5:]]
        elif len(class_0_values) > 5: example_values = [class_0_values[:5], class_1_values]
        elif len(class_1_values) > 5: example_values = [class_0_values, class_1_values[-5:]]
        else: example_values = [class_0_values, class_1_values]

        return render(request, 'agents/validate_anomaly.html', {"username": request.user.username, "agent": agent, "dataset": dataset, "counts": counts, "examples": example_values, "mse": mse_values})
