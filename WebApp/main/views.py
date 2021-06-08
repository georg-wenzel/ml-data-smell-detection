from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from main.utility.DjangoUtility import dict_contains_all
from .models import Agent, Dataset, Column, Smell
from main.utility.DatabaseUtility import safe_get
from main.utility import GensimUtility
from main.utility.AgentUtility import agent_handlers
from django.core.cache import cache
from main.forms import AddAgentForm
import pandas as pd
from django.utils.html import escape

#views for non dataset- and agent-specific calls

#render base logged in page if logged in, otherwise redirect to user login
def home(request):
    if request.user.is_authenticated:
        smells = Smell.objects.all
        form = AddAgentForm()
        return render(request, 'main/baseloggedin.html', {"username": request.user.username, "smells": smells, "form": form})
    else:
        return redirect('/user/login')

#analyze page (validate trained agent against uploaded dataset)
def analyze(request):
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #if not post request
    if request.method != "POST":
        #get all agents for this user that have a settings file or have an external training set (i.e. are trained)
        agents = [x for x in Agent.objects.filter(user=request.user) if x.settings.name or x.external_prepared]
        #get all datasets for this user
        datasets = Dataset.objects.filter(user=request.user)
        #display in form
        return render(request, 'main/analyze.html', {"username": request.user.username, "agents": agents, "datasets": datasets})

    #if post
    if request.method == "POST":
        #kwargs will contain arguments which are not required for every type of agent
        kwargs = dict()

        #make sure all required parameters are passed (agent and dataset)
        if not dict_contains_all(request.POST, ["dataset", "agent"]):
            messages.add_message(request, messages.ERROR, "Not all required fields were filled.")
            return redirect('/analyze')

        dataset = safe_get(Dataset, id=request.POST['dataset'])
        agent = safe_get(Agent, id=request.POST['agent'])

        if not agent or not dataset:
            messages.add_message(request, messages.ERROR, "Could not find agent or dataset.")
            return redirect('/analyze/')

        #make sure column is present for non-dedupe agents    
        if 'column' in request.POST:
            column = safe_get(Column, id=request.POST['column'])
            if not column:
                messages.add_message(request, messages.ERROR, "Could not find specified column.")
                return redirect('/analyze/')
            kwargs['column'] = column
        else:
            if agent.agent_type.id != 1: #if not dedupe agent, error
                messages.add_message(request, messages.ERROR, "Column must be specified for non-Dedupe agents.")
                return redirect('/analyze') 
        
        #make sure rcsSum, rcsThreshold and rcsSecondaryAgent are present for gensim agents
        if dict_contains_all(request.POST, ['rcsThreshold', 'rcsNum', 'rcsSecondaryAgent']):
            kwargs['rcsNum'] = int(request.POST['rcsNum'])
            kwargs['rcsThreshold'] = float(request.POST['rcsThreshold'])
            if(request.POST['rcsSecondaryAgent'] != '-1'): kwargs['rcsSecondaryAgent'] = int(request.POST['rcsSecondaryAgent'])
        else:
            if agent.agent_type.id == 2:
                messages.add_message(request, messages.ERROR, "RCS value and threshold must be specified for gensim agents.")
                return redirect('/analyze') 

        #pass to agent based handling if we got this far
        return agent_handlers[agent.agent_type.id-1].analyze(request, agent, dataset, **kwargs)

#duplicates page (called via post from gensim analyze screen to give details on synonymous pairs of words)
def duplicates(request):
    #authentication
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #only allow post requests
    if request.method != "POST":
        messages.add_message(request, messages.ERROR, "This link must be called via POST.")
        return redirect('/analyze')

    #get dataset and column provided in the form
    if not dict_contains_all(request.POST, ["dataset", "column"]):
        messages.add_message(request, messages.ERROR, "Not all required fields were provided.")
        return redirect('/analyze')

    dataset = safe_get(Dataset, id=request.POST['dataset'])
    column = safe_get(Column, id=request.POST['column'])
    if not column or not dataset or not (dataset.user == request.user) or not (column.dataset == dataset):
        messages.add_message(request, messages.ERROR, "An error occured during authorization or fetching the requested data.")
        return redirect('/analyze')

    #each pair key is in the form word1,word2,rcs
    #out of this key, we build the corresponding word pair
    pairs = []
    for key in request.POST:
        if len(key.split(",")) == 3:
            word1 = key.split(",")[0]
            word2 = key.split(",")[1]
            rcs = float(key.split(",")[2])
            #append the word pair
            pairs.append((word1, word2, rcs))
    
    # this is simply dictionary preparation to avoid key errors
    # the dictionary occurence has keys in order of indices
    # at each index is another dictionary containing "word1", "word2" with the respective word of the word pair
    # and "word1_occurences" and "word2_occurences" containing every sentence in the corresponding column which contains this word.
    # finally, we simply propagate the RCS again
    # word1_examples and word2_examples contain up to 5 examples of escaped strings with a <strong> tag around the desired word.
    # These are not shown in the newly created dataset, but passed on as examples to the template.
    occurences = dict()
    for i, (word1, word2, rcs) in enumerate(pairs):
        occurences[i] = dict()
        occurences[i]["word1"] = word1
        occurences[i]["word2"] = word2
        occurences[i]["word1_occurences"] = []
        occurences[i]["word2_occurences"] = []
        occurences[i]["rcs"] = rcs
        occurences[i]["word1_examples"] = []
        occurences[i]["word2_examples"] = []

    #here we get the sentences in an in-memory array
    sentences = GensimUtility.get_unfiltered(dataset.upload.path, column.name)

    #we iterate over the sentences and find all word occurences and add them to the corresponding dictionary entry
    for sentence in sentences:
        for i, (word1, word2, _) in enumerate(pairs):
            if " " + word1 + " " in sentence:
                occurences[i]["word1_occurences"].append(sentence)
                if len(occurences[i]["word1_examples"]) < 5:
                    occurences[i]["word1_examples"].append(escape(sentence).replace(word1, "<strong>" + word1 + "</strong>"))
            elif " " + word2 + " " in sentence:
                occurences[i]["word2_occurences"].append(sentence)
                if len(occurences[i]["word2_examples"]) < 5:
                    occurences[i]["word2_examples"].append(escape(sentence).replace(word2, "<strong>" + word2 + "</strong>"))
    
    rows = []
    #iterate over all word pairs and append occurences as well as ratio to each word pair, build new dataset
    for i in occurences:
        rows.append([occurences[i]["word1"], occurences[i]["word2"], occurences[i]["rcs"], len(occurences[i]["word1_occurences"]), 
                        len(occurences[i]["word2_occurences"]), 
                        round(len(occurences[i]["word1_occurences"])/len(occurences[i]["word2_occurences"]), 4) if len(occurences[i]["word2_occurences"]) > 0 else None])


    #store new dataset in cache
    df = pd.DataFrame(rows, columns=["word", "synonym", "rcs", "Word 1 occurences", "Word 2 occurences", "Occurence Ratio"])
    cache.set(request.user.username + "_dataset", df)

    #display the results
    return render(request, 'main/duplicates.html', {"dataset": dataset, "column": column, "matches": occurences, "username": request.user.username})

#view which is called when the user clicks a download button
def download(request):    
    #authentication
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #get the cache value for this user
    csv = cache.get(request.user.username + "_dataset")

    #check that the cache file exists and is a pandas dataframe
    if (type(csv) == pd.DataFrame):
            # return this object as a filestream
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment;filename="results.csv"'
            csv.to_csv(response, index=False)
            return response

    #otherwise return to the analyze page with an error
    messages.add_message(request, messages.ERROR, "No file to download was found for this user.")
    return redirect('/analyze')

#view which is called when the user requests to download an agent's model file
def download_model(request, id):
    #authentication
    if not request.user.is_authenticated:
        return redirect('/user/login')

    agent = safe_get(Agent, id=id)

    #make sure agent exists and belongs to this user
    if not agent:
        messages.add_message(request, messages.ERROR, ERR_UNAUTHORIZED.format("agent"))
        return redirect('/agents')
    if agent.user != request.user:
        messages.add_message(request, messages.ERROR, ERR_UNAUTHORIZED.format("agent"))
        return redirect('/agents')

    #make sure the agent has a model file
    if not agent.model:
        messages.add_message(request, messages.ERROR, "Requested agent does not have a model file.")
        return redirect('/agents/' + id)

    # return the file as a filestream
    with open(agent.model.name, 'rb') as f:
        response = HttpResponse(f.read())
        response['Content-Disposition'] = 'attachment;filename="model"'
        return response

#view which is called when the user requests to download an agent's settings file
def download_settings(request, id):
    #authentication
    if not request.user.is_authenticated:
        return redirect('/user/login')

    agent = safe_get(Agent, id=id)

    #make sure agent exists and belongs to this user
    if not agent:
        messages.add_message(request, messages.ERROR, ERR_UNAUTHORIZED.format("agent"))
        return redirect('/agents')
    if agent.user != request.user:
        messages.add_message(request, messages.ERROR, ERR_UNAUTHORIZED.format("agent"))
        return redirect('/agents')

    #make sure the agent has a settings file
    if not agent.settings:
        messages.add_message(request, messages.ERROR, "Requested agent does not have a settings file.")
        return redirect('/agents/' + id)

    # return the file as a filestream
    with open(agent.settings.name, 'rb') as f:
        response = HttpResponse(f.read())
        response['Content-Disposition'] = 'attachment;filename="settings"'
        return response
