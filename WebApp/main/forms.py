from django import forms
from django.forms import modelformset_factory
from crispy_forms.helper import FormHelper
from .models import Dataset, Agent, Column

### DATASET FORMS
#Form for adding a dataset
class AddDatasetForm(forms.ModelForm):
    name = forms.CharField(label="Dataset Name")
    description = forms.CharField(label="Dataset Description (optional)", required=False)
    upload = forms.FileField(label="Dataset File", help_text="At this point, only .csv files are supported.")
    has_headers = forms.BooleanField(label="Has header?", required=False, help_text="If checked, will automatically attempt to read the column names from the uploaded file.")

    class Meta:
        model = Dataset
        fields = ["name", "description", "upload"]

#Form for editing  a dataset
class EditDatasetForm(forms.ModelForm):
    name = forms.CharField(label="Dataset Name")
    description = forms.CharField(label="Dataset Description (optional)", required=False)

    class Meta:
        model = Dataset
        fields = ["name", "description"]

#Form for editing a column
class EditColumnForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_tag = False
        self.helper.template = 'bootstrap/table_inline_formset_alt.html'

    name = forms.CharField(label="Column Name")
    dtype = forms.CharField(label="Column Data Type")

    class Meta:
        model = Column
        fields = ["name", "dtype"]
#Formset
EditColumnFormSet = modelformset_factory(Column, form=EditColumnForm, extra=0)

#Form for deleting a dataset
class DeleteDatasetForm(forms.Form):
    id = forms.IntegerField()

### AGENT FORMS
# Form for adding an agent
class AddAgentForm(forms.ModelForm):
    name = forms.CharField(label="Agent Name")
    description = forms.CharField(label="Agent Description (optional)", required=False)
    agent_type = forms.IntegerField(widget = forms.HiddenInput())

    class Meta:
        model = Agent
        fields = ["name", "description"]

# Form for editing an agent (basic info)
class EditAgentForm(forms.ModelForm):
    name = forms.CharField(label="Agent Name")
    description = forms.CharField(label="Agent Description (optional)", required=False)

    class Meta:
        model = Agent
        fields = ["name", "description"]

#Form for deleting an agent
class DeleteAgentForm(forms.Form):
    id = forms.IntegerField()
