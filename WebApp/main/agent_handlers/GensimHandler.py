from django.shortcuts import render, redirect
import tempfile
import os
from main.models import Dataset, Agent, AgentColumn, Column
from main.utility import GensimUtility
from main.utility import AgentUtility
import numpy as np
from main.utility.DjangoUtility import dict_contains_all
from main.utility import StringUtility
from gensim.models import Word2Vec
from gensim.parsing.porter import PorterStemmer
import gensim.downloader
from main.utility.DatabaseUtility import safe_get
from django.core.files.base import File
import pandas as pd
from django.core.cache import cache
from django.contrib import messages

# contains logic for setting up, training and analyzing data using a gensim agent

class GensimHandler:

    # Called when user clicks edit agent the first time
    @staticmethod
    def setup(request, agent):

        # if not a POST request, open the corresponding setup page (selecting dataset and column)
        if request.method != "POST":
            # get datasets from this user
            datasets = Dataset.objects.filter(user=request.user)
            # get pretrained models made available by gensim
            gensim_pretrained = gensim.downloader.info()['models']
            # define licenses which are open source, as defined by gensim downloader repository
            whitelisted_licenses = [
                "http://opendatacommons.org/licenses/pddl/",
                "https://github.com/commonsense/conceptnet-numberbatch/blob/master/LICENSE.txt",
                "not found"
            ]
            #get valid models
            models = [k for k in gensim_pretrained.keys() if "license" in gensim_pretrained[k] and gensim_pretrained[k]['license'] in whitelisted_licenses]

            return render(request, 'agents/setup_gensim.html', {"username": request.user.username, "agent": agent, "datasets": datasets, "pretrained": models})

        # if post request, check if a valid dataset and column id are in the POST data
        if dict_contains_all(request.POST, ["selftrained-setup", "dataset", "columnId"]):
            dataset = safe_get(Dataset, id=int(request.POST["dataset"]))
            column = safe_get(Column, id=int(request.POST["columnId"]))
            if not column:
                messages.add_message(request, messages.ERROR,
                                     StringUtility.ERR_INVALID_KEY.format("dataset"))
                return redirect('/agents')
            if not dataset:
                messages.add_message(request, messages.ERROR,
                                     StringUtility.ERR_INVALID_KEY.format("column"))
                return redirect('/agents')

            # make sure column is part of the dataset & save this dataset and column
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
                ac = AgentColumn(name=column.name, dtype="String", agent=agent)
                ac.save()
            else:
                messages.add_message(request, messages.ERROR,
                                     StringUtility.ERR_INVALID_KEY.format("column"))
            return redirect('/agents/train/' + str(agent.id))
        # if not both column and dataset are present, check if a pretrained set key is given
        elif dict_contains_all(request.POST, ["pretrainedSet", "pretrained-setup"]):
            pretrained = request.POST["pretrainedSet"]

            if not pretrained in gensim.downloader.info()['models'].keys():
                messages.add_message(request, messages.ERROR,
                                     StringUtility.ERR_INVALID_KEY.format("pretrained dataset"))
                return redirect('/agents')

            agent.external_set_key = pretrained
            agent.save()
            return redirect('/agents/train/' + str(agent.id))
        # if a pretrained set key is not given either, send an error
        else:
            messages.add_message(
                request, messages.ERROR, "Neither a valid dataset and column nor a valid pretrained dataset were selected.")
            return redirect('/agents/train/' + str(agent.id))

    # called when user clicks edit agent on subsequent times (after setting up)
    @staticmethod
    def train(request, agent):
        # check to see if this agent is already in the session
        if not request.session.get('agent_id', False) == agent.id:
            # if not, set it to this agents id
            request.session["agent_id"] = agent.id

            #either training on the selected dataset, or downloading the external dataset
            if agent.dataset:
                # get dataset and column
                column = safe_get(AgentColumn, agent=agent)

                #tokenize sentences
                tokenized_sentences = GensimUtility.get_corpus(
                    agent.dataset.upload.path, column.name)
                #train word2vec model
                model = Word2Vec(sentences=tokenized_sentences,
                                sg=0, vector_size=300, window=4, epochs=100)
                #create a temporary file to store the model into
                temppath = os.path.join(tempfile.gettempdir(), next(
                    tempfile._get_candidate_names()))
                model.save(temppath)
                readfile = open(temppath, 'rb')
                #store to both model and settings as we only have one file to store for this type of agent
                agent.model.save("", File(readfile), save=True)
                agent.settings.save("", File(readfile), save=True)
                agent.iterations = agent.iterations + 1
                agent.save()
                readfile.close()
            elif agent.external_set_key:
                # download the model to make sure it's available later
                gensim.downloader.load(agent.external_set_key)
                #this flag lets us know that we can display the agent as an option on the analyze page
                agent.external_prepared = True
                agent.iterations = agent.iterations + 1
                agent.save()

        messages.add_message(
            request, messages.SUCCESS, "Your agent has been trained successfully.")
        return redirect('/agents')

    # called when user uses this agent on the analyze page
    @staticmethod
    def analyze(request, agent, dataset, **kwargs):
        # make sure column is of type string
        mismatches = AgentUtility.dataset_all_columns_match_unopened(dataset.upload.name, 
                            [(kwargs['column'].name, np.object)])
        if len(mismatches) > 0:
            messages.add_message(
                request, messages.ERROR, StringUtility.ERR_COLUMN_MISMATCH(mismatches))
            return redirect('/analyze')
            
        #if everything is ok, load word vectors
        # if training dataset is external, load word vectors from gensim downloader
        if agent.external_prepared:
            wv = gensim.downloader.load(agent.external_set_key)
        # otherwise load word vectors from saved file
        else:
            wv = Word2Vec.load(agent.settings.path).wv

        # repeat the same thing for the secondary dataset if rcsSecondaryAgent was given to the method as kw argument
        wv_secondary = None
        if 'rcsSecondaryAgent' in kwargs:
            secondary_agent = safe_get(
                Agent, id=int(kwargs['rcsSecondaryAgent']))

            #this requires some more validation because this input is not sanity checked by the main url handler
            if secondary_agent:
                if secondary_agent.user != request.user:
                    messages.add_message(
                        request, messages.ERROR, StringUtility.ERR_UNAUTHORIZED.format("secondary agent"))
                    return redirect('/agents')

                if secondary_agent.agent_type.id != 2:
                    #if the agent is not gensim
                    messages.add_message(
                        request, messages.ERROR, StringUtility.ERR_INVALID_KEY.format("secondary agent"))
                    return redirect('/agents')

                if secondary_agent.external_prepared:
                    wv_secondary = gensim.downloader.load(
                        secondary_agent.external_set_key)
                elif secondary_agent.settings:
                    wv_secondary = Word2Vec.load(
                        secondary_agent.settings.path).wv
                else:
                    #this happens when the secondary agent has not at least been trained once
                    messages.add_message(
                        request, messages.ERROR, StringUtility.ERR_INVALID_KEY.format("secondary agent"))
                    return redirect('/agents')
            else:
                #this happens when the key given in the form was not a valid agent
                messages.add_message(
                        request, messages.ERROR, StringUtility.ERR_INVALID_KEY.format("secondary agent"))
                return redirect('/agents')

        # get the vocabulary by building a word2vec instance, but not training it (this ensures the vocabulary is built in the same way)
        vocab = GensimUtility.get_vocabulary(dataset.upload.path, kwargs['column'].name)

        has_secondary = wv_secondary is not None

        #decrease vocabulary size by only taking the intersection of all 2 (or 3) word sets
        vocab = set(vocab).intersection(set(wv.key_to_index))
        if has_secondary: vocab = vocab.intersection(set(wv_secondary.key_to_index))

        pairs = []
        # iterate over all words in the vocabulary
        for i, key in enumerate(vocab):
            # check the top 3 most similar words in the trained space
            for word, sim in wv.similar_by_word(key, topn=5):
                # skip ones that aren't part of the vocab we are testing against
                if not (word in vocab):
                    continue
                # otherwise calculate the relative cosine similarity compared to the top n similar words
                rel = wv.relative_cosine_similarity(
                    key, word, topn=kwargs['rcsNum'])
                # if there is a secondary set of wordvectors and both words are also in it, add that score on top
                if has_secondary:
                    rel += wv_secondary.relative_cosine_similarity(
                        key, word, topn=kwargs['rcsNum'])
                # use the threshold given by the form
                if rel > kwargs['rcsThreshold']:
                    # if relativity > threshold, add this as a "synonymous pair"
                    pairs.append((key, word, rel))
                else:
                    # if the relativity isnt > 0.11, then the relativity for the following pairs cannot be > 0.11 either, so we can break this loop
                    break

        #sort pairs by rcs
        pairs = sorted(pairs, key=lambda x: -x[2])

        #for building the downloadable csv file
        syn_word1 = []
        syn_word2 = []
        syn_rcs = []

        #use a porter stemmer to mostly avoid adding inflections to the pairs
        #also removes duplicate pairs
        uniquepairs = []
        pairlist = []
        porter = PorterStemmer()
        for idx, (word, word2, rel) in enumerate(pairs):
            #avoids adding inflections of words
            if (not (porter.stem(word) == porter.stem(word2)) and
                #avoids adding duplicate pairs with lower rcs
                not ((word2, word) in pairlist)):
                    uniquepairs.append((word, word2, round(rel, 4),
                                    word + ',' + word2 + ',' + str(round(rel,4))))
                    pairlist.append((word, word2))
                    syn_word1.append(word)
                    syn_word2.append(word2)
                    syn_rcs.append(rel)

        #store dataset in cache
        df = pd.DataFrame({"word":syn_word1, "synonym":syn_word2, "rcs": syn_rcs})
        cache.set(request.user.username + "_dataset", df)

        # warning that is displayed if there is too many pairs
        # the dataset for download always contains the full list, but the web-app only shows the top n
        entrywarning = False
        if len(uniquepairs) > 10000:
            uniquepairs = uniquepairs[:10000]
            entrywarning = True

        return render(request, 'agents/validate_gensim.html', {"username": request.user.username, "agent": agent, "dataset": dataset, "column": kwargs['column'], "pairs": uniquepairs,
                        "entrywarning": entrywarning})
