from django.shortcuts import render, redirect
from main.models import Dataset, Agent, AgentColumn, Column
from main.utility import DedupeUtility
from main.utility import StringUtility
from main.utility import AgentUtility 
import pandas as pd
import numpy as np
from main.utility.DatabaseUtility import safe_get
from django.contrib import messages
import dedupe
from django.core.cache import cache
from django.core.files.base import File

# contains logic for setting up, training and analyzing data using a Dedupe Agent

class DedupeHandler:

    # Called when user clicks edit agent the first time
    @staticmethod
    def setup(request, agent):

        # if not a POST request, open the corresponding setup page (selecting a dataset and column to train on)
        if request.method != "POST":
            datasets = Dataset.objects.filter(user=request.user)
            return render(request, 'agents/setup_dedupe.html', {"username": request.user.username, "agent": agent, "datasets": datasets})

        # if post request, store the selected dataset and columns in the agent
        if not ("dataset" in request.POST):
            messages.add_message(request, messages.ERROR,
                                 StringUtility.ERR_MISSING_KEY.format("dataset"))
            return redirect('/agents')

        # get dataset
        dataset = safe_get(Dataset, id=request.POST["dataset"])
        if not dataset:
            messages.add_message(request, messages.ERROR,
                                 StringUtility.ERR_INVALID_KEY.format("dataset"))
            return redirect('/agents')

        #open dataset for type checking

        # get columns
        columns = []
        for key in request.POST:
            if key.startswith("check"):
                col = Column.objects.get(
                    id=int(request.POST[key]), dataset=dataset)
                columns.append(col)
        if len(columns) == 0:
            messages.add_message(
                request, messages.ERROR, "At least one column must be provided to train on.")
            return redirect('/agents')

        # make sure all columns are part of the dataset
        if set(columns).issubset(set(Column.objects.filter(dataset=dataset))):

            # make sure all columns are of type string
            mismatches = AgentUtility.dataset_all_columns_match_unopened(dataset.upload.name, 
                            list(map(lambda x: (x.name, np.object), columns)))
            if len(mismatches) > 0:
                messages.add_message(
                    request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
                return redirect('/agents')

            # save dataset and columns
            agent.dataset = dataset
            agent.save()
            for c in columns:
                # all dedupe fields are treated as strings because dedupe only works by string distances, not through comparing numeric values.
                ac = AgentColumn(name=c.name, dtype="String", agent=agent)
                ac.save()
            return redirect('/agents/train/' + str(agent.id))

        else:
            messages.add_message(
                request, messages.ERROR, "Not all columns were part of the selected dataset.")
        return redirect('/agents')

    # called when user clicks edit agent on subsequent times (after setting up)
    @staticmethod
    def train(request, agent):
        loading_error = False

        # check to see if this dedupe agent was the last agent clicked
        if not request.session.get('agent_id', False) == agent.id:
            # if not, set it to this agents id
            request.session["agent_id"] = agent.id
            # reset the agent stored in cache
            cache.delete(request.user.username + "_agent")

            # get all columns from this agent
            columns = AgentColumn.objects.filter(agent=agent)
            # create a dedupe model using these columns
            model = dedupe.Dedupe(DedupeUtility.set_fields(columns), num_cores=0)
            data = DedupeUtility.read_data(agent.dataset.upload.path)
            # if this agent has been trained before
            if agent.model.name:
                # use the existing training data and iterate
                f = open(agent.model.path, 'r')
                model.prepare_training(data, training_file=f)
                f.close()
            else:
                # otherwise train from scratch
                model.prepare_training(data)

            # store this model in the cache for active labelling
            cache.set(request.user.username + "_agent", model)

        # check to see if there is a Dedupe agent in cache for this user
        model = cache.get(request.user.username + "_agent")
        # if the cached agent is a dedupe model, assume it is the correct one
        if not isinstance(model, dedupe.api.Dedupe):
            loading_error = True

        if not loading_error:
            # show active labelling screen
            model = cache.get(request.user.username + "_agent")
            return render(request, 'agents/train_dedupe.html', {"username": request.user.username, "agent": agent, "data": model.uncertain_pairs()})
        else:
            messages.add_message(
                request, messages.ERROR, "An error occured trying to load the agent from the cache system.")
            return redirect('/agents')


    # called when user uses this agent on the analyze page
    @staticmethod
    def analyze(request, agent, dataset, **kwargs):
        #get data in the form needed for dedupe to process
        data = DedupeUtility.read_data(dataset.upload.path)
        #get dataset as pandas dataframe to add a cluster column
        data_csv = pd.read_csv(dataset.upload.path)
        agent_cols = AgentColumn.objects.filter(agent = agent)

        # make sure all columns are of type string
        mismatches = AgentUtility.dataset_all_columns_match(data_csv, 
                        list(map(lambda x: (x.name, np.object), agent_cols)))
        if len(mismatches) > 0:
            messages.add_message(
                request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
            return redirect('/analyze')

        with open(agent.settings.path, 'rb') as f:
            # setup StaticDedupe instance using the settings file (which contains the latest iteration of stored weights)
            model = dedupe.StaticDedupe(f, num_cores=0)
            
            # cluster samples
            clusters = model.partition(data)

            # column which assigns a label (cluster id) to each row
            cluster_label = [0] * data_csv.shape[0]
            # column which assigns the cluster probability to each row
            cluster_probability = [0] * data_csv.shape[0]

            # prepare columns for the dataset download as well as the web-app display
            clusters_full = []
            for i, (records, scores) in enumerate(clusters, start=1):
                cluster = []
                for j, record in enumerate(records):
                    #assign a cluster id to the label column
                    cluster_label[record] = i
                    cluster_probability[record] = scores[j]
                    cluster.append(data[record])
                # do not show clusters with 1 entry (unique data)
                if len(cluster) > 1:
                    #get all variants
                    variants = []
                    for record in cluster:
                        entry = []
                        for col in agent_cols:
                            entry.append(record[col.name])
                        variants.append(tuple(entry))
                    variants_set = set(variants)
                    #skip clusters with only one variant (true duplicates, do not constitue a data smell)
                    if len(variants_set) < 2:
                        continue

                    occurences = list(map(lambda x: variants.count(x), variants_set))

                    clusters_full.append(list(zip(variants, occurences)))

            # store labelled dataset in cache 
            
            data_csv['cluster'] = cluster_label
            data_csv['probability'] = cluster_probability
            cache.set(request.user.username + "_dataset", data_csv)

        return render(request, 'agents/validate_dedupe.html', {"username": request.user.username, "agent": agent, "dataset": dataset, "data": data, "clusters": clusters_full})
