from django.shortcuts import redirect
from main.models import Dataset, Agent, Column, Smell
from django.http import HttpResponseBadRequest, JsonResponse
from django.core.cache import cache
from main.utility.DatabaseUtility import safe_get
from main.utility import DedupeUtility
from main.utility import StringUtility
import json
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import File
from django.contrib import messages

# Views returned by Ajax calls
# Will generally return a JsonResponse containing data or an HttpResponse containing an error code and text.
# I recommend not building the return object inside the JsonResponse() call as this has lead to errors for me before.

# view which returns all columns from a dataset if the user is authorized to view it.
# id is the id of the dataset to get the columns of
@csrf_exempt
def get_columns(request, id):
    if not request.user.is_authenticated:
        return HttpResponseBadRequest("Must be logged in to get column names.")

    dataset = safe_get(Dataset, id=id)
    if not dataset:
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Dataset"))
    if not (dataset.user == request.user):
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Dataset"))

    columns = []
    for col in Column.objects.filter(dataset=dataset):
        columns.append({'id': col.id, 'name': col.name, 'dtype': col.dtype})

    res = {'columns': columns}
    return JsonResponse(res)


# view which returns the type of agent (integer) for a given agent if the user is authorized to view it.
# id is the id of the agent to get the type of
@csrf_exempt
def get_agent_type(request, id):
    if not request.user.is_authenticated:
        return HttpResponseBadRequest("Must be logged in to get agent type.")

    agent = safe_get(Agent, id=id)
    if not agent:
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Agent"))
    if not (agent.user == request.user):
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Agent"))

    res = {'type': agent.agent_type.id}
    return JsonResponse(res)

# view which returns data as json for a given smell ID
# id is the id of the smell to get the data of
@csrf_exempt
def get_smell_data(request, id):

    smell = safe_get(Smell, id=id)
    if not smell:
        return HttpResponseBadRequest(StringUtility.ERR_INVALID_KEY.format("Smell"))

    res = {'name': smell.name, 'description': smell.description, 'agent_type_id': smell.agent_type.id, 
            'agent_type_name': smell.agent_type.name, 'dataset_description': smell.dataset_description}
    return JsonResponse(res)

# view which returns the id and name of all gensim agents the user has access to, that do not have the id given in the request
# this is used to provide a list of options for using two agents to build the summed rcs of
# id is the id of the gensim agent to exclude
@csrf_exempt
def get_other_gensim_agents(request, id):
    if not request.user.is_authenticated:
        return HttpResponseBadRequest("Must be logged in to get agents.")

    agents = Agent.objects.filter(agent_type__id = 2, user = request.user).exclude(id = id)

    res = {'agents': [{'id': a.id, 'name': a.name} for a in agents]}
    return JsonResponse(res)

# view which returns a dedupe pair for the user to label
# the corresponding agent needs to be stored in cache for this to work
@csrf_exempt
def get_dedupe_pair(request):
    if not request.user.is_authenticated:
        return HttpResponseBadRequest("Must be logged in to start the training process")
    try:
        # load model from cache
        model_dedupe = cache.get(request.user.username + "_agent")
        # post request means the user has also submitted a duplicate pair, so we get this from the post request and mark it
        if request.method == "POST": 
            # convert arrays to tuples
            pairs = json.loads(request.body)
            distinct = list(map(lambda x: tuple(x), pairs["distinct"]))
            match = list(map(lambda x: tuple(x), pairs["match"]))
            #mark pairs
            model_dedupe.mark_pairs({"match": match, "distinct": distinct})
        # get a new set of uncertain pairs
        uncertain = model_dedupe.uncertain_pairs()
        # store agent back to the cache
        cache.set(request.user.username + "_agent", model_dedupe)
        # return new uncertain pairs
        return JsonResponse({"pairs": uncertain})
    # This probably means that the model loaded does not have a get_uncertain_pairs() method, i.e. no dedupe agent is stored properly
    except AttributeError:
        return HttpResponseBadRequest("Agent cached is not of proper type. (Dedupe)")


# view which stores the current model saved in cache, to the agent id stored in the session
# on success, redirects the user to the agents page.
@csrf_exempt
def store_dedupe_training(request):
    if not request.user.is_authenticated:
        return HttpResponseBadRequest("Must be logged in to store the training process.")

    agent = safe_get(Agent, id=request.session.get("agent_id", None))
    if not agent:
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Agent"))
    if agent.agent_type.id != 1:
        return HttpResponseBadRequest("Agent in session is not of proper type. (Dedupe)")
    if not (agent.user == request.user):
        return HttpResponseBadRequest(StringUtility.ERR_UNAUTHORIZED.format("Agent"))
        
    try:
        # load model from cache
        model = cache.get(request.user.username + "_agent")
        # train and get files
        model_file, settings_file = DedupeUtility.train_model(model)
        
        #store files
        readfile = open(model_file, 'r')
        agent.model.save("", File(readfile), save=True)
        readfile.close()
        readfile = open(settings_file, 'rb')
        agent.settings.save("", File(readfile), save=True)
        readfile.close()

        # store increased agent iteration and save
        agent.iterations = agent.iterations + 1
        agent.save()

        messages.add_message(
            request, messages.SUCCESS, "Your agent has been trained successfully.")

        # redirect to main page
        return redirect('/agents')
    # This probably means that the model loaded does not have a get_uncertain_pairs() method, i.e. no dedupe agent is stored properly
    except AttributeError:
        return HttpResponseBadRequest("Model cached is not of proper type. (Dedupe)")
