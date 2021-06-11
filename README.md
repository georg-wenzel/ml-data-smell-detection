# Table of Contents
1. [Important Constraints](#constraints)
2. [Folder Structure](#folders)
3. [Setup Guide](#setupguide)  
3.1 [Scripts](#scripts)  
3.2 [Web-App Option 1: Docker](#docker)  
3.3 [Web-App Option 2: Dockerless](#dockerless)  
4. [How to use the software](#howtouse)  
4.1 [Agent Outputs](#outputs)


# Important Constraints (Please Read) <a name="constraints"></a>
**This section clears up possibly unexpected behavior when using the application. I strongly recommend reading it.**

The web application has been tested on Windows using **Python 3.6**. I **strongly** recommend using Python 3.6, as under this specific version and with the provided requirements file, there is no compatibility issues. On other Python versions, different NumPy version requirements from TensorFlow and Dedupe may lead to crashes on startup of the web application server.

The application is subject to several constraints when it comes to parallelization:
- Using this application in parallel (i.e., with multiple users at once), may lead to performance issues caused by the training of agents or analysis of datasets.
- Undefined behavior may occur when opening the application in multiple tabs as the same user. Please follow a serial workflow per user.
- Per user, only the **last validated dataset** will be downloaded using any download button, as the generated dataset is stored in a cache. This is **by design**, as to avoid having to re-run the agent when the download button is pressed (which could take considerable time), or storing the file everytime the agent is ran. 
    - This means, if you validate a dataset using one agent, then validate another, then click the download button of the first agent, the second dataset will be downloaded.

Essentially, consider your workflow serial for predictable results. Training and validating agents may take multiple minutes, depending on agent type and dataset size.  

Datasets uploaded must be in .csv format and UTF-8 encoded.  

An active internet connection is required to download pre-trained gensim models (this is also the case for the pre-defined sample gensim agents).  

The account login data for the root user, containing sample datasets and agents, is as follows:
```
username: root
password: root
```
Keep in mind that while these agents model the agents evaluated in the thesis itself, they are not the same agents, and as such may perform slightly better or worse. For example, the LSTM agent within this implementation seems to be able to better classify dates with spelled out months than the agent evaluated within the thesis. However, agents performing slightly worse in some regards is also a possibility.  

This implementation is meant to be used on a private network. If you wish to make this application publically available, consider following the Django deployment checklist (https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/) to avoid any security complications.
# ML Data Smell Analysis

## Folder Structure<a name="folders"></a>
`Scripts/` contains the scripts used for synthetic dataset generation, as well as additional scripts used to extract data from the results of the agents. Generally, `data-generation.py` in each folder is used to generate the synthetic datasets, while `results-classification.py` in each folder is used to further analyze the results of the classification done in the web-application. Each script starts with a detailed documentation on how to use it, so please refer to these comments for further information.

`WebApp/` contains the main Django application and follows the general Django convention.  
`WebApp/thesis` is the main starting point of the Django application and defines the general settings and modules.  
`WebApp/user` is a small app providing the default login and registration endpoints of the Django Auth system.  
`WebApp/ajax` is an app returning JsonResponses to specific requests made via Ajax by JavaScript functions.  
`WebApp/main` is the main app accepting calls in the WebApp and manages most calls to the WebApp.  
`WebApp/main/agent_handlers` is where the main logic for setting up, training and using ML agents is written.  
`WebApp/main/utility` is a module where utility functions are refactored to, to make other modules more legible.  
`WebApp/data` is the file storage point for datasets and models uploaded by all users. The root user already has a certain amount of datasets and agents predefined. If this is undesired, please delete them from within the application rather than in the folder.

## Setup Guide<a name="setupguide"></a>
### Scripts<a name="scripts"></a>
Any script can simply be ran as a python file using `python [filename]`. Interactive input will handle the rest. Before starting a script, navigate to the `Scripts/` directory and run `pip install -r requirements.txt` (optionally, in a virtual environment) to make sure all required packages are installed.

### Web-App Option 1: Docker<a name="docker"></a>
A docker-compose file and Dockerfile is provided. If docker is installed, you can use the following installation process.
1. Navigate to the `WebApp/` directory
2. Rename `WebApp/db_sample.sqlite3`to `WebApp/db.sqlite3`
3. Run `docker-compose up`

### Web-App Option 2: Dockerless<a name="dockerless"></a>
To start the WebApp without Docker, take the following steps. 

1. Navigate to the `WebApp/` directory
2. Rename `WebApp/db_sample.sqlite3`to `WebApp/db.sqlite3`
3. Run `pip install -r requirements.txt`
4. **Important: Find your Django library files and implement the Django Fix detailed below. The WebApp is not able to store Dedupe instances and large datasets in-memory otherwise!!**
5. Download [memcached](https://memcached.org/), or download [memcached-windows](https://github.com/jefyt/memcached-windows) for Windows.
7. Run memcached on the default port with a 10mb memory limit using `memcached -I 10m`
8. Run `python manage.py runserver` in the `WebApp/` directory.

**Django Fix:**
Django's memcached interface does not by default respect an increased per-entry memory limit specified in settings.py or the memcached instance, which leads to large variables not being stored. Thus, it is necessary to go into your django package files under `core/cache/backends/memcached.py` and add the following line:
```python
class MemcachedCache(BaseMemcachedCache):
    "An implementation of a cache binding using python-memcached"
    def __init__(self, server, params):
        import memcache
>       memcache.SERVER_MAX_VALUE_LENGTH = 10485760 #add this line to accept 10mb cache entries
        super().__init__(server, params, library=memcache,
        value_not_found_exception=ValueError)
```
See: https://stackoverflow.com/a/15383495/2941598  
Alternatively, replace the aforementioned file with the file `WebApp/django-memcached-fix.py`.

## How to use the software<a name="howtouse"></a>
By default, the software runs on the default Django port. After following the setup guide, it can be reached on `localhost:8000`. You will be prompted to login. Create a new user or use the default username/password stored in the sample database (root/root). This root user additionally has access to predefined training datasets and agents.

The WebApp is written to be as self descriptive as possible. Below is a quick explanation of a possible workflow.  

1. Upload a Dataset (.csv File). Use the "Has Header" checkbox in the upload form to automatically infer column names or manually set them in the "Edit Dataset" window.
2. Create an agent of the type you wish to use for analysis. Click the "train" button to define training dataset and column, and follow the steps on the screen. Agents other than the Dedupe agent are unsupervised, meaning you simply have to wait. In the agent overview, the agent will show up as "Trained" once training has been complete. If the agent has already been trained before, the iterations will increase. **Do not train multiple agents at once**. Most agents use a large amount of CPU resources to train, and some rely on per-user in-memory data.
3. When your agent is trained, go to the "Analyze" page and select a test dataset, validation agent, and other parameters, if necessary. After clicking "Validate Dataset", results will be displayed.
4. Interpret the results, as detailed below.
5. If desired, you may download a labeled version of the dataset for further analysis.

### Agent Outputs<a name="outputs"></a>
**Dedupe**  
The WebApp outputs duplicate clusters which have more than one entry in their set (i.e. no one-entry clusters, no clusters where all entries are made up of the exact same values). If trained correctly, these clusters correspond to potential smell instances. The downloadable dataset appends a "cluster" and "probability" column to your uploaded dataset. Same cluster IDs imply the entries are in the same cluster. The probability is a measure of how sure Dedupe is that the samples belong in the same cluster.  

**Gensim**  
The WebApp outputs all word pairs with a relative cosine similarity greater than the given threshold. The downloadable dataset contains these word pairs and their corresponding rcs. When word pairs are manually filtered (through checkboxes in the WebApp), occurrence data is appended (i.e. how often each word in the pair occurs in the dataset as a standalone word).

**TensorFlow (LSTM)**  
The WebApp outputs class distribution metrics (how much of the data is in each class) and some example entries for each class. The downloadable dataset contains the uploaded dataset with a class and probability score for each sample.

**TensorFlow (Anomaly Detection)**  
The WebApp outputs class distribution metrics (how much of the data is above the threshold) and the lowest and highest reconstruction error samples. The downloadable dataset contains the uploaded dataset with the reconstruction error and a class value (i.e. is the reconstruction error greater than the threshold).
