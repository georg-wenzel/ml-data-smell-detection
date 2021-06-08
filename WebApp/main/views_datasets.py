from django.shortcuts import render, redirect
from main.forms import AddDatasetForm, EditDatasetForm, DeleteDatasetForm, EditColumnFormSet
from django.http import HttpResponseBadRequest
from main.utility.DatabaseUtility import safe_get
from main.utility import StringUtility
from main.models import Dataset, Column, Agent
from django import forms
import pandas
from django.contrib import messages

# views for dataset-specific calls

# view which lists all datasets
def datasets(request):
    if not request.user.is_authenticated:
        return redirect('/user/login')
        
    # Create form for adding and deleting a dataset
    form = AddDatasetForm()
    delete_form = DeleteDatasetForm()
    # add a hidden id field to the dataset deletion form 
    # (this will be filled when a user clicks the button on any specific dataset)
    delete_form.fields['id'].widget = forms.HiddenInput()
    datasets = Dataset.objects.filter(user=request.user)
    return render(request, 'main/datasets.html', {"username": request.user.username, 
                    "form": form, "datasets": datasets, "deleteForm": delete_form})

# view which creates a new dataset 
def add_dataset(request):
    # authentication
    if not request.user.is_authenticated:
        return redirect('/user/login')

    # must be submitted as post
    if request.method != "POST":
        return redirect('/datasets')

    #get the data from the post request into an AddDatasetForm
    form = AddDatasetForm(request.POST, request.FILES)

    #validate data, add user and store if file is valid
    if(form.is_valid):
        #ignore files that are not content type text/csv
        if not request.FILES['upload'].name.endswith(".csv"):
            messages.add_message(request, messages.ERROR, "Only .csv files are supported for uploading.")
            return redirect('/datasets')

        #store but dont commit
        dataset = form.save(commit=False)
    
        #set user to authenticated user
        dataset.user = request.user
        #open csv file
        try:
            csv = pandas.read_csv(dataset.upload, index_col=False)
            dataset.row_length = csv.shape[0]
            dataset.col_length = csv.shape[1]
            dataset.save()

            #if user checked the "has headers", attempt to extract column types and headers from the dataset
            if form.cleaned_data['has_headers']:
                for name, dtype in csv.dtypes.iteritems():
                    c = Column(name=name,dtype=dtype,dataset=dataset)
                    c.save()
            #otherwise, store them as undefined
            else:
                for _ in range(csv.shape[1]):
                    c = Column(name="undefined",dtype="undefined",dataset=dataset)
                    c.save()

            #redirect to dataset list
            return redirect('/datasets')
        except SystemExit:
            raise
        except:
            messages.add_message(request, messages.ERROR, "There was an error parsing your .csv file.")
            return redirect('/datasets')
    
# view which edits an existing dataset / displays details
def edit_dataset(request, id=1):
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #get dataset with the given id
    dataset = safe_get(Dataset, id=id)

    if not dataset:
        messages.add_message(request, messages.ERROR, "Could not find dataset.")
        return redirect('/datasets')

    #check if user is authorized to see this dataset
    if not dataset.user == request.user:
        messages.add_message(request, messages.ERROR, StringUtility.ERR_UNAUTHORIZED("dataset"))
        return redirect('/datasets')

    #if not post request
    if request.method != "POST":
        #provide an edit form (and edit column form set) and return the view
        form = EditDatasetForm(instance=dataset, prefix='dataset')
        formset = EditColumnFormSet(queryset=Column.objects.filter(dataset=dataset), prefix='columns')

        return render(request, 'main/editdataset.html', {"username": request.user.username, 
                            "dataset": dataset, "form": form, "colforms": formset})


    #if post request, get the data from the form and formset
    form = EditDatasetForm(request.POST, instance=dataset, prefix='dataset')
    formset = EditColumnFormSet(request.POST, prefix='columns')

    #update form and formset if valid
    if form.is_valid:
        form.save()
    if formset.is_valid:
        instances = formset.save(commit=False)
        for f in instances:
            f.dataset = dataset
            f.save()
        for f in formset.deleted_objects:
            f.delete()

    #return to datasets view
    return redirect('/datasets')

# view which deletes a dataset
def delete_dataset(request):
    #must be called via post
    if request.method != "POST":
        return redirect('/datasets')

    #must be called by logged in user
    if not request.user.is_authenticated:
        return redirect('/user/login')

    #get dataset with this id
    form = DeleteDatasetForm(request.POST)
    #delete if dataset exists and owner is equal to user calling the POST
    if form.is_valid():
        dataset = safe_get(Dataset, pk=form.cleaned_data['id'])
        if dataset:
            if(dataset.user == request.user):
                #we can only delete the dataset if it has not been used in the training of agents
                agent_datasets = Agent.objects.filter(dataset = dataset)
                if len(agent_datasets) == 0:
                    dataset.upload.delete()
                    dataset.delete()
                else:
                    #build error string
                    err = "This dataset is used to train the following agents: " +  (", ".join([a.name for a in agent_datasets]))
                    err += ". Please first remove the corresponding agents."
                    #show error
                    messages.add_message(request, messages.ERROR, err)
    
    #either way, redirect to the datasets page afterwards
    return redirect('/datasets')