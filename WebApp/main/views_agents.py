from django.shortcuts import render, redirect
from main.forms import AddAgentForm, EditAgentForm, DeleteAgentForm
from django.http import HttpResponseBadRequest
from main.models import Agent, AgentType, Dataset, AgentColumn
from django import forms
from main.utility.DatabaseUtility import safe_get
from main.utility.AgentUtility import agent_handlers
import pandas
from django.contrib import messages

# views for agent-specific calls

# view which lists all datasets
def agents(request):
    if not request.user.is_authenticated:
        return redirect('/user/login')
    #get all agents of this user and prepare the agent deletion form
    agents = Agent.objects.filter(user=request.user)
    form = DeleteAgentForm()
    
    # add a hidden id field to the agent deletion form 
    # (this will be filled when a user clicks the button on any specific agent)
    form.fields['id'].widget = forms.HiddenInput()

    #if the user renders this page, reset the session key for which agent was last clicked (user has left the training page)
    if "agent_id" in request.session: del request.session["agent_id"]

    return render(request, 'main/agents.html', {"username": request.user.username, "agents": agents, "deleteForm": form})

# view which adds an agent
def add_agent(request):
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #if not post request
    if request.method != "POST":
        #populate the add agent form
        form = AddAgentForm()
        #set the initial agent type to 1
        form.fields['agent_type'].initial = 1
        return render(request, 'main/addagent.html', {"username": request.user.username, "form": form})

    # if post, populate form
    form = AddAgentForm(request.POST)

    # validate data, add user
    if(form.is_valid()):
        atype = safe_get(AgentType, id=form.cleaned_data['agent_type'])
        if atype:
            # store but dont commit
            agent = form.save(commit=False)
            # set user to authenticated user
            agent.user = request.user
            agent.agent_type = atype
            agent = agent.save()
    else:
        messages.add_message(request, messages.ERROR, "An unknown error occured while creating this agent.")

    return redirect('/agents')

# view for training an agent
def train_agent(request, id):
    if not request.user.is_authenticated:
        return redirect('/user/login')

    # get agent with this id
    agent = safe_get(Agent, id=id)
    if not agent:
        messages.add_message(request, messages.ERROR, "Could not find agent.")
        return redirect('/agents')

    # check if user is authorized to see this agent
    if not agent.user == request.user:
        messages.add_message(request, messages.ERROR, StringUtility.ERR_UNAUTHORIZED("agent"))
        return redirect('/agents')

    # method of training the agent depends on the type of agent, so defer to corresponding method
    # within each agent handler

    # if the agent does not have a training dataset set yet, defer to setup render
    if not agent.dataset and not agent.external_set_key:
        return agent_handlers[agent.agent_type.id-1].setup(request, agent)
    # if it does, defer to training render
    else:
        return agent_handlers[agent.agent_type.id-1].train(request, agent)


# view for editing an agent
def edit_agent(request, id=1):
    if not request.user.is_authenticated:
        return redirect('/user/login')

    # get agent with this id
    agent = safe_get(Agent, id=id)
    if not agent:
        messages.add_message(request, messages.ERROR, "Could not find agent.")
        return redirect('/agents')

    # check if user is authorized to see this agent
    if not agent.user == request.user:
        messages.add_message(request, messages.ERROR, StringUtility.ERR_UNAUTHORIZED("agent"))
        return redirect('/agents')

    # if not post request
    if request.method != "POST":
        #prepare editing forms
        form = EditAgentForm(instance=agent)
        #prepare list of columns the agent was trained on
        columns = AgentColumn.objects.filter(agent=agent)
        cols = ", ".join([x.name + " (" + x.dtype + ")" for x in columns])
        #render info
        return render(request, 'main/editagent.html', {"username": request.user.username, "agent": agent,
                        "form": form, "columns": cols})

    # if post request
    # update agent data if valid
    form = EditAgentForm(request.POST, instance=agent)
    if form.is_valid:
        form.save()
        return redirect('/agents')

# view to delete an agent
def delete_agent(request):
    # must be called with post 
    if request.method != "POST":
        return redirect('/agents')

    if not request.user.is_authenticated:
        return redirect('/user/login')

    form = DeleteAgentForm(request.POST)
    if form.is_valid():
        # get agent with this id, delete if everything is valid
        agent = safe_get(Agent, pk=form.cleaned_data['id'])
        if agent:
            if(agent.user == request.user):
                agent.model.delete()
                agent.settings.delete()
                agent.delete()

    # redirect to agents view
    return redirect('/agents')
