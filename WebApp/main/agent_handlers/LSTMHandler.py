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

# contains logic for setting up, training and analyzing data using a TensorFlow agent (LSTM)

class LSTMHandler:

    # Called when user clicks edit agent the first time
    @staticmethod
    def setup(request, agent):

        # if not a POST request, open the corresponding setup page (selecting a dataset and column to train on)
        if request.method != "POST":
            datasets = Dataset.objects.filter(user=request.user)
            return render(request, 'agents/setup_tensorflow.html', {"username": request.user.username, "agent": agent, "datasets": datasets})
        
        #otherwise get dataset, x and y column
        if not dict_contains_all(request.POST, ["dataset", "columnId", "labelColumnId"]):
            messages.add_message(request, messages.ERROR,
                        "Not all form data was provided.")
            return redirect('/agents')

        dataset = safe_get(Dataset, id=int(request.POST["dataset"]))
        column = safe_get(Column, id=int(request.POST["columnId"]))
        label = safe_get(Column, id=int(request.POST["labelColumnId"]))
        if(not dataset or not column or not label):
            messages.add_message(request, messages.ERROR,
                    "Form data provided was not valid.")
            return redirect('/agents')

        # make sure columns are part of the dataset & save this dataset and columns
        if column.dataset == dataset and label.dataset == dataset:

            # make sure columns are of expected datatype
            mismatches = AgentUtility.dataset_all_columns_match_unopened(dataset.upload.name, 
                            [(column.name, np.object),(label.name, np.int64)])
            if len(mismatches) > 0:
                messages.add_message(
                    request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
                return redirect('/agents')

            agent.dataset = dataset
            agent.save()
            # store column for x as String, column for y as int
            ac = AgentColumn(name=column.name, dtype="String", agent=agent)
            ac.save()
            lc = AgentColumn(name=label.name, dtype="int", agent=agent)
            lc.save()
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
            # prepare training for model
        dataset = safe_get(Dataset, id=agent.dataset.id)
        # get all agent columns
        columns = AgentColumn.objects.filter(agent=agent)
        # the column with type "String" is x, the column with type "int" is y
        x_col = next(x for x in columns if x.dtype == "String")
        y_col = next(y for y in columns if y.dtype == "int")

        if not dataset or not x_col or not y_col:            
            messages.add_message(request, messages.ERROR,
                "Failed to load x and y columns for this dataset.")
            return redirect('/agent')

        # get x and y data from the dataset
        csv = TensorflowUtility.get_csv(dataset.upload.path)
        x = csv[x_col.name]
        y = csv[y_col.name]
        # encode x and y columns
        encoder = TensorflowUtility.get_encoder(x)
        enc_x = TensorflowUtility.encode_x(encoder, x)
        enc_y = TensorflowUtility.encode_y(y)
        # get class weights and train model
        weights = TensorflowUtility.get_class_weights(enc_y)
        model = TensorflowUtility.train_model(encoder, enc_x, enc_y, weights)
        # store model in model, encoder in settings
        temppath = os.path.join(tempfile.gettempdir(), next(
            tempfile._get_candidate_names()) + ".keras")
        model.save(temppath)
        temp = open(temppath, 'rb')
        agent.model.save("", File(temp), save=True)
        temp.close()
        #use pickle to serialize the encoder so we can in the future encode data using the same encoder
        f = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump(encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        # load encoder from settings file (deserialize)
        with open(agent.settings.path, mode='rb') as f:
            encoder = pickle.load(f)
        #get x
        x = data[kwargs['column'].name]
        # encode data
        enc_x = TensorflowUtility.encode_x(encoder, x)
        # classify data
        y = model.predict(enc_x)

        classes = [[] for _ in range(len(y[0]))]

        #labels and probability for .csv file export
        csv_labels = [0] * data.shape[0]
        csv_probability = [0] * data.shape[0]

        # group by class
        for i in range(len(y)):
            class_x = np.argmax(y[i])
            percent = str(round(y[i][class_x] * 100, 2))
            classes[class_x].append((x[i], percent))

            csv_labels[i] = class_x
            csv_probability[i] = percent

        # compute classwise data (Ratio) and examples to show
        ratio = []
        total_samples = []
        examples = []
        for i in range(y.shape[1]):
            total_samples.append(len(classes[i]))
            ratio.append(round(len(classes[i])/x.shape[0] * 100, 2))
            if len(classes[i]) > 5:
                examples.append(classes[i][:5])
            else:
                examples.append(classes[i])

        # store dataset in cache
        data['label'] = csv_labels
        data['probability'] = csv_probability
        cache.set(request.user.username + "_dataset", data)

        return render(request, 'agents/validate_tensorflow.html', {"username": request.user.username, "agent": agent, "dataset": dataset, 
                        "class_data": list(zip(ratio, examples, total_samples))})
