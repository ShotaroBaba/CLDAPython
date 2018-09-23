# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:45:47 2018

@author: Shotaro Baba
"""
# Reading all necessary packages

# Import all necessary packages.import sys
import sys
sys.path.append("../models/")
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    
from nltk.corpus import wordnet as wn
import CLDA_eval_screen
import CLDA
import gc
import numpy as np


from functools import partial
import tkinter as tk
import os
import time
import xml.etree.ElementTree as ET
import pandas as pd 
from tkinter.filedialog import askdirectory
import csv
import pickle
import itertools
import json

from multiprocessing import cpu_count
from multiprocessing import Pool
# from multiprocessing import Process


import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

import asyncio
import concurrent.futures
import requests

lemmatizer = WordNetLemmatizer()

# Create default values
# for experiment
dataset_dir = "../../CLDA_data_training"
dataset_test = "../../CLDA_data_testing"
concept_prob_suffix_json = "_c_prob.json"
concept_name_suffix_txt = "_c_name.txt"
feature_matrix_suffix_csv = "_f_mat.csv"
feature_name_suffix_txt = "_f_name.txt"
file_name_df_suffix_csv = "_data.csv"
CLDA_suffix_pickle = "_CLDA.pkl"
LDA_suffix_pickle = "_LDA.pkl"
label_text_suffix = ".txt"

converted_xml_suffix = "_conv.txt"
converted_folder = "converted_xml_files"

stop_word_folder = "../stopwords"
stop_word_smart_txt = "smart_stopword.txt"


#Define smart stopwords
smart_stopwords = []
with open(stop_word_folder + '/' + stop_word_smart_txt , "r") as f:
    for line in f:
    #Remove the \n
        smart_stopwords.append(line.strip('\n'))

default_K = 10
default_smooth = 0.0001 
default_ngram = 1
default_min_df = 0.02
default_max_df = 0.98
default_topic_num = 5
default_max_iter = 20

label_delim = " "
delim = ","
#K = 0
#with open( "../../CLDA_data_testing" + '/' + "ForTest_c_prob.json", "r") as f:
#    K = json.load(f)
#
# Download all necessary nltk download
# components
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# Create main menu            
# Defining the lemmatizer



gc.enable()
core_use = 5

######################################################
#######Define stop words here...
######################################################
def define_sw():
    return set(stopwords.words('english') + smart_stopwords)
def lemmatize(token, tag):
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(tag[0], wordnet.NOUN)

    return lemmatizer.lemmatize(token, tag)

# The tokenizer for the documents
def cab_tokenizer(document):
    tokens = []
    sw = define_sw()
    punct = set(punctuation)

    # split the document into sentences
    for sent in sent_tokenize(document):
        # tokenize each sentence
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # If stopword, ignore token and continue
            if token in sw:
                continue

            # Lemmatize the token and add back to the token
            lemma = lemmatize(token, tag)

            # Append lemmatized token to list
            tokens.append(lemma)
    return tokens



# Create vectorise files
# Define the function here
# Generate vector for creating the data
def generate_vector(ngram_length, min_df, max_df):
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=[ngram_length[0], ngram_length[1]],
                           min_df=min_df, max_df=max_df)

# Generate count vectorizer
def vectorize(tf_vectorizer, df):
    # Generate_vector
    # df = df.reindex(columns=['text'])  # reindex on tweet

    tf_matrix = tf_vectorizer.fit_transform(df['Text'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    return tf_matrix, tf_feature_names
######################################################
####### End of stop word definition
######################################################


# Retrieve the text file from the files
def generate_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield (path + '\\' + file)

# Returns texts in the xml documents 
def return_text(path, path_string, x):
    file = path + '/' + path_string + '/' +str(x) + ".xml"
    tree = ET.parse(file)
    root = tree.getroot()
    result = ''
    for element in root.iter():
        if(element.text != None):
            result += element.text + ' '
    result = result[:-1]
    
    with open(file + ".txt", "w", encoding = 'utf-8') as f:
        f.write(result)
        
    return result

# Read all files in the training dataset


# Read the test files for the test purpose
def read_test_files():
    for_test_purpose_data = pd.DataFrame([], columns=['File', 'Text'])
    training_path = []

    # Creating 
    test_path = "../../R8-Dataset/Dataset/ForTest"
    for dirpath, dirs, files in os.walk(test_path):
        training_path.extend(files)
    
    # Remove the files other than xml files
    training_path = [x for x in training_path if x.endswith('xml')]

    # Remove the path where the 
    # Extract only last directory name
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        
        # training_data_list.append(path_string)
        # Initialise the list of the strings
        # for_test_purpose_data[path_string] = {}
        
        file = test_path + '/' + path_string 
        tree = ET.parse(file)
        
        # Turn the string in xml into text
        root = tree.getroot()
        result = ''
        # Retrieving the text from xml files
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        result = result[:-1] # Removing the space at the back of the string
        
                
        # Initialise 
        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']))
    
    return for_test_purpose_data


# Asynchoronic feature vector retrieval
def create_feature_vector_async(file_string, feature_list, dataset_dir, ngram,  min_df, max_df):
    
    file_string = os.path.splitext(os.path.basename(file_string))[0]
    if any(file_string in substring for substring in feature_list):
        print("Feature {} already exists".format(file_string +  feature_matrix_suffix_csv))
        return None
    else:
        # Ensure load just in case...
        wn.ensure_loaded()
        # Read csv files 
        print(file_string)
        datum = pd.read_csv(dataset_dir + '/' + file_string + file_name_df_suffix_csv, encoding='utf-8', sep=',', 
                    error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
        time.sleep(3)
        # Vectorise the document 
        vect = generate_vector(ngram, min_df, max_df)
        vectorized_data, feature_names = vectorize(vect, datum)
        
        # Save array as the csv file
        np.savetxt(dataset_dir + '/' + file_string + feature_matrix_suffix_csv, vectorized_data.toarray(), delimiter = delim)
        
#            with open(dataset_dir + '/' + file_string + '.csv')
        with open(dataset_dir + '/' + file_string + feature_name_suffix_txt, "w") as f:
            for i in feature_names:
                f.write("{}\n".format(i))
        

        return file_string + feature_matrix_suffix_csv

# Later the value K, and smooth will allowed to be changed 
# in GUI to generate CLDA
# Used for lighten the burden of the calculation
def create_concept_matrix_async(file_string, concept_list, dataset_dir, K = default_K, smooth = default_smooth):
        
        async def retrieve_word_concept_data(feature_names):
            with concurrent.futures.ThreadPoolExecutor(max_workers=150) as executor:
                collection_of_results = []
                with requests.Session() as s:
                    loop = asyncio.get_event_loop()
                    futures = [
                        loop.run_in_executor(
                            executor, 
                            s.get, 
                            'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' +
                            i.replace(' ', '+') + 
                            '&topK=' + str(K) + # This K will later be adjustable in later stage...
                            '&smooth=' + str(smooth) # The value is smoothen by the given value
                        )
                        for i in feature_names
                    ]
                    for response in await asyncio.gather(*futures):
                        collection_of_results.append(response.json())
                
                    return collection_of_results
        
        file_string = os.path.splitext(os.path.basename(file_string))[0]
        # Check whether the test subject exists or not
        if any(file_string in substring for substring in concept_list):
            # Feature already exists
            print("Feature {} already exists".format(file_string + concept_prob_suffix_json))
        else:
            p_e_c  = {}

            feature_names = []                    
            with open(dataset_dir + '/' + file_string + feature_name_suffix_txt, "r") as f:
                for line in f:
                    feature_names.append(line.strip('\n'))
            
            
            #Sort the feature names just in case...
            feature_names = sorted(feature_names)
            '''
            # Retrieve the tenth rankings of the words
            # K needed to be adjustable so that the
            # Researcher can find the characteristics of
            # all values!
            '''
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(retrieve_word_concept_data(feature_names))
            results = loop.run_until_complete(future)
            
            # temporary
            for idx, i  in enumerate(feature_names):

                p_e_c[i] = results[int(idx)]
            
            # List up the concept names
            
            l = [list(i.keys()) for i in list(p_e_c.values())]
            concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
            
            #    concept_sets[len(concept_sets)-1]
            # Put the atom concept if there are no concepts in the words
            
            
            # Adding atomic elements
            for i in feature_names:
                #if there are no concepts in the words, then...
                if p_e_c[i] == {}:
                    
                    # Append the words with no related concpets
                    # as the atomic concepts
                    concept_names.append(i)
                    
            # Sorting the concept_names after adding feature names
            concept_names = sorted(concept_names)

            # Save p(e|c) as json file
            with open(dataset_dir + '/' + file_string + concept_prob_suffix_json, "w") as f:
                json.dump(p_e_c, f, indent = "\t")
            
            # Save concept_names as text file
            with open(dataset_dir + '/' + file_string + concept_name_suffix_txt, "w") as f:
                for i in concept_names:
                    f.write("{}\n".format(i))
                    
            
            return file_string + concept_prob_suffix_json


class Application():
    
        
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("File Pre Processing & Model Creation")

        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="Quit", command= self.root.destroy)
        self.menubar.add_command(label="Help", command=None)
        
        self.root.config(menu=self.menubar)
        
        self.start_menu()
        
        '''
        Training part
        '''
        # The list of the files for the upload
        self.upload_file_list_labelling = []
        self.upload_file_list_classifier = []
        
        self.linked_folders = []
        self.folder_directory = None
        self.document_data = []
        self.labelling_dir = ""
        
        '''
        List all LDA and CLDA list to
        the values
        '''
        self.folder_name_list = []
        
        
        self.LDA_model_list = []
        
        # The generation of CLDA model list
        # Later this will be used for generating results
        
        self.CLDA_model_list = []
        self.LDA_model_list = []
        
        # Initialisation of the topic feature concept list by retrieving the data from
        # the documents
        
        '''
        Testing part
        '''
       
        # The list of the files for the upload
        self.upload_file_list_labelling_test = []
        self.upload_file_list_classifier_test = []
        
        
        '''
        List all LDA and CLDA  to
        the values
        '''
        self.folder_name_list_test = []
        
        
        self.LDA_model_list_test = []
        
        # The generation of CLDA model list
        # Later this will be used for generating results
        self.CLDA_model_list_test = []
        self.LDA_model_list_test = []
        
        
        
        
        self.retrieve_topic_feature_concept_list()
        
        self.root.mainloop()
        
        
    def start_menu(self):
        
        # Bring all the frame which include all the Listbox frame together
        '''
        ##############################################################################
        ###Main frame
        ##############################################################################
        '''
        self.main_group = tk.Frame(self.root)
        self.main_group.pack()
        
        self.main_listbox_and_buttons = tk.Frame(self.main_group, relief = tk.RAISED, borderwidth = 1)
        self.main_listbox_and_buttons.pack(side = tk.LEFT, padx = 5, pady = 5)
        
        self.result_box = tk.Frame(self.main_group)
        self.result_box.pack(side = tk.RIGHT, anchor = tk.N)
        
        self.main_listbox_and_result = tk.Frame(self.main_listbox_and_buttons)
        self.main_listbox_and_result.pack()
        
        '''
        #######################################
        ####Folder selection & feature generation part
        #######################################
        '''
        
        # Main frame of the selection
        self.frame_for_folder_selection = tk.Frame(self.main_listbox_and_result)
        self.frame_for_folder_selection.pack(side = tk.LEFT)
        
        self.user_drop_down_select_folder_label = tk.Label(self.frame_for_folder_selection , text = "Folder (Topic)\nSelection")
        self.user_drop_down_select_folder_label.pack()
        
        self.frame_for_drop_down_menu_folder = tk.Frame(self.frame_for_folder_selection)
        self.frame_for_drop_down_menu_folder.pack()
        
        self.scrollbar_drop_down_menu_file = tk.Scrollbar(self.frame_for_drop_down_menu_folder, orient = "vertical")
        self.scrollbar_drop_down_menu_file.pack(side = tk.RIGHT, fill = 'y')
        
        self.user_drop_down_select_folder_list = tk.Listbox(self.frame_for_drop_down_menu_folder, exportselection=0)
        self.user_drop_down_select_folder_list.pack(side = tk.LEFT, fill = 'y')
        self.user_drop_down_select_folder_list['yscrollcommand'] = \
        self.scrollbar_drop_down_menu_file.set
        self.scrollbar_drop_down_menu_file['command'] = self.user_drop_down_select_folder_list.yview
        
        '''
        #######################################
        ####Folder selection & feature generation part end
        #######################################
        '''
        
        '''
        #######################################
        ####Result screen
        #######################################
        '''
        self.folder_selection_result_frame = tk.Frame(self.result_box)
        self.folder_selection_result_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.user_drop_down_file_selection_results_label = tk.Label(self.folder_selection_result_frame,
                                                                    text = "Selection & Processing Result")
        
        self.user_drop_down_file_selection_results_label.pack(side = tk.TOP) 
        
        self.user_drop_down_folder_selection_results_frame = tk.Frame(self.folder_selection_result_frame)
        self.user_drop_down_folder_selection_results_frame.pack()
        
        
        self.user_drop_down_folder_selection_results_scroll_bar = \
        tk.Scrollbar(self.user_drop_down_folder_selection_results_frame, orient = "vertical")
        self.user_drop_down_folder_selection_results_scroll_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.user_drop_down_folder_selection_results_scroll_list = tk.Text(self.user_drop_down_folder_selection_results_frame)
        self.user_drop_down_folder_selection_results_scroll_list.pack(side = tk.LEFT, fill = 'y')
        self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
        
        self.user_drop_down_folder_selection_results_scroll_list['yscrollcommand'] = \
        self.user_drop_down_folder_selection_results_scroll_bar.set
        
        self.user_drop_down_folder_selection_results_scroll_bar['command'] = \
        self.user_drop_down_folder_selection_results_scroll_list.yview
        '''
        #######################################
        ####Result screen END
        #######################################
        '''
        
        
        '''
        #######################################
        ####Word vector generation part
        ####Already created list part
        #######################################
        '''
        self.word_vector_generation_list_frame = tk.Frame(self.main_listbox_and_result)
        self.word_vector_generation_list_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_list_word_vector_label = tk.Label(self.word_vector_generation_list_frame,
                                                         text = "Word Vector\nCreated List")
        self.drop_down_list_word_vector_label.pack()
        
        self.drop_down_list_word_vector_frame = tk.Frame(self.word_vector_generation_list_frame)
        self.drop_down_list_word_vector_frame.pack()
        
        self.drop_down_list_word_vector_list = tk.Listbox(self.drop_down_list_word_vector_frame,
                                                          exportselection = 0)
        
        self.drop_down_list_word_vector_list.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_list_word_vector_bar = tk.Scrollbar(self.drop_down_list_word_vector_frame, orient = "vertical")
        self.drop_down_list_word_vector_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_list_word_vector_list['yscrollcommand'] = \
        self.drop_down_list_word_vector_bar.set
        
        self.drop_down_list_word_vector_bar['command'] = \
        self.drop_down_list_word_vector_list.yview
        
        '''
        #######################################
        ####Word vector generation part end
        #######################################
        '''
        
        '''
        #######################################
        ####Concept word probabilities creation part
        ####
        #######################################
        '''
        self.concept_prob_generation_list_frame = tk.Frame(self.main_listbox_and_result)
        self.concept_prob_generation_list_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_concept_prob_vector_label = tk.Label(self.concept_prob_generation_list_frame,
                                                         text = "Concept Prob\nCreated List")
        self.drop_down_concept_prob_vector_label.pack()
        
        self.drop_down_concept_prob_vector_frame = tk.Frame(self.concept_prob_generation_list_frame)
        self.drop_down_concept_prob_vector_frame.pack()
        
        self.drop_down_concept_prob_vector_list = tk.Listbox(self.drop_down_concept_prob_vector_frame,
                                                          exportselection = 0)
        
        self.drop_down_concept_prob_vector_list.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_concept_prob_vector_bar = tk.Scrollbar(self.drop_down_concept_prob_vector_frame, orient = "vertical")
        self.drop_down_concept_prob_vector_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_concept_prob_vector_list['yscrollcommand'] = \
        self.drop_down_concept_prob_vector_bar.set
        
        self.drop_down_concept_prob_vector_bar['command'] = \
        self.drop_down_concept_prob_vector_list.yview
        
        
        '''
        #######################################
        ####Show the list of pre-existed CLDA model
        #######################################
        '''
        self.CLDA_generation_list_frame = tk.Frame(self.main_listbox_and_result)
        self.CLDA_generation_list_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_CLDA_label = tk.Label(self.CLDA_generation_list_frame,
                                                         text = "CLDA\nCreated List")
        self.drop_down_CLDA_label.pack()
        
        self.drop_down_CLDA_frame = tk.Frame(self.CLDA_generation_list_frame)
        self.drop_down_CLDA_frame.pack()
        
        self.drop_down_CLDA_list = tk.Listbox(self.drop_down_CLDA_frame,
                                                          exportselection = 0)
        
        self.drop_down_CLDA_list.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_CLDA_bar = tk.Scrollbar(self.drop_down_CLDA_frame, orient = "vertical")
        self.drop_down_CLDA_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_CLDA_list['yscrollcommand'] = \
        self.drop_down_CLDA_bar.set
        
        self.drop_down_CLDA_bar['command'] = \
        self.drop_down_CLDA_list.yview
        
        '''
        #######################################
        ####Show the list of pre-existed LDA model
        #######################################
        '''
        self.LDA_generation_list_frame = tk.Frame(self.main_listbox_and_result)
        self.LDA_generation_list_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_LDA_label = tk.Label(self.LDA_generation_list_frame,
                                                         text = "LDA\nCreated List")
        self.drop_down_LDA_label.pack()
        
        self.drop_down_LDA_frame = tk.Frame(self.LDA_generation_list_frame)
        self.drop_down_LDA_frame.pack()
        
        self.drop_down_LDA_list = tk.Listbox(self.drop_down_LDA_frame,
                                                          exportselection = 0)
        
        self.drop_down_LDA_list.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_LDA_bar = tk.Scrollbar(self.drop_down_LDA_frame, orient = "vertical")
        self.drop_down_LDA_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_LDA_list['yscrollcommand'] = \
        self.drop_down_LDA_bar.set
        
        self.drop_down_LDA_bar['command'] = \
        self.drop_down_LDA_list.yview
        
        
        
        '''
        #######################################################################################
        ###Retrieving the test data part 
        #######################################################################################

        '''
        self.main_listbox_and_result_test = tk.Frame(self.main_listbox_and_buttons)
        self.main_listbox_and_result_test.pack()
        
        
       
        '''
        #######################################
        ####Folder selection (test)
        #######################################
        '''
        
        self.frame_for_folder_selection_test = tk.Frame(self.main_listbox_and_result_test )
        self.frame_for_folder_selection_test.pack(side = tk.LEFT)
        
        self.user_drop_down_select_folder_label_test = tk.Label(self.frame_for_folder_selection_test , text = "Created \nTest Data")
        self.user_drop_down_select_folder_label_test.pack()
        
        self.frame_for_drop_down_menu_folder_test = tk.Frame(self.frame_for_folder_selection_test)
        self.frame_for_drop_down_menu_folder_test.pack()
        
        self.scrollbar_drop_down_menu_file_test = tk.Scrollbar(self.frame_for_drop_down_menu_folder_test, orient = "vertical")
        self.scrollbar_drop_down_menu_file_test.pack(side = tk.RIGHT, fill = 'y')
        
        self.user_drop_down_select_folder_list_test = tk.Listbox(self.frame_for_drop_down_menu_folder_test, exportselection=0)
        self.user_drop_down_select_folder_list_test.pack(side = tk.LEFT)
        self.user_drop_down_select_folder_list_test['yscrollcommand'] = \
        self.scrollbar_drop_down_menu_file_test.set
        self.scrollbar_drop_down_menu_file_test['command'] = self.user_drop_down_select_folder_list_test.yview
        

        
        '''
        #######################################
        ####Feature vector extraction
        #######################################
        '''
        
#        self.word_vector_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
#        self.word_vector_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
#        
#        self.drop_down_list_word_vector_label_test = tk.Label(self.word_vector_generation_list_frame_test,
#                                                         text = "Word Vector\nCreated List (Test)")
#        self.drop_down_list_word_vector_label_test.pack()
#        
#        self.drop_down_list_word_vector_frame_test = tk.Frame(self.word_vector_generation_list_frame_test)
#        self.drop_down_list_word_vector_frame_test.pack()
#        
#        self.drop_down_list_word_vector_list_test = tk.Listbox(self.drop_down_list_word_vector_frame_test,
#                                                          exportselection = 0)
#        
#        self.drop_down_list_word_vector_list_test.pack(side = tk.LEFT, fill = 'y')
#        
#        self.drop_down_list_word_vector_bar_test = tk.Scrollbar(self.drop_down_list_word_vector_frame_test, orient = "vertical")
#        self.drop_down_list_word_vector_bar_test.pack(side = tk.RIGHT, fill = 'y')
#        
#        self.drop_down_list_word_vector_list_test['yscrollcommand'] = \
#        self.drop_down_list_word_vector_bar_test.set
#        
#        self.drop_down_list_word_vector_bar_test['command'] = \
#        self.drop_down_list_word_vector_list_test.yview
    
        
        '''
        #######################################
        ####Concpet-word vector extraction
        #######################################
        '''
#        
#        self.concept_prob_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
#        self.concept_prob_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
#        
#        self.drop_down_concept_prob_vector_label_test = tk.Label(self.concept_prob_generation_list_frame_test,
#                                                         text = "Concept Prob\nCreated List (Test)")
#        self.drop_down_concept_prob_vector_label_test.pack()
#        
#        self.drop_down_concept_prob_vector_frame_test = tk.Frame(self.concept_prob_generation_list_frame_test)
#        self.drop_down_concept_prob_vector_frame_test.pack()
#        
#        self.drop_down_concept_prob_vector_list_test = tk.Listbox(self.drop_down_concept_prob_vector_frame_test,
#                                                          exportselection = 0)
#        
#        self.drop_down_concept_prob_vector_list_test.pack(side = tk.LEFT, fill = 'y')
#        
#        self.drop_down_concept_prob_vector_bar_test = tk.Scrollbar(self.drop_down_concept_prob_vector_frame_test, orient = "vertical")
#        self.drop_down_concept_prob_vector_bar_test.pack(side = tk.RIGHT, fill = 'y')
#        
#        self.drop_down_concept_prob_vector_list_test['yscrollcommand'] = \
#        self.drop_down_concept_prob_vector_bar_test.set
#        
#        self.drop_down_concept_prob_vector_bar_test['command'] = \
#        self.drop_down_concept_prob_vector_list_test.yview
        
        
        '''
        ################################################################################
        ###End of retrieving the test data part 
        ################################################################################
        '''
        
        # Forming the button group for putting them together
        self.values_to_input = tk.Frame(self.main_listbox_and_buttons)
        self.values_to_input.pack()
        
        
        self.top_concept_label = tk.Label(self.values_to_input, text = "Top concept to retrieve (positive integer): ")
        self.top_concept_label.grid(row = 0, column =0)
        
        self.top_concept_text = tk.Entry(self.values_to_input)
        self.top_concept_text.grid(row = 0, column = 1)
        
        
        self.smooth_label = tk.Label(self.values_to_input, text = "Smooth value (float): ")
        self.smooth_label.grid(row = 0, column = 2)
        
        self.smooth_text = tk.Entry(self.values_to_input)
        self.smooth_text.grid(row =0, column = 3)
        
        self.ngram_label = tk.Label(self.values_to_input, text = "min_ngram value (positive integer): ")
        self.ngram_label.grid(row = 1, column = 0)
        
        self.ngram_text = tk.Entry(self.values_to_input)
        self.ngram_text.grid(row = 1, column = 1)
        
        self.ngram_max_label = tk.Label(self.values_to_input, text = "max_ngram value (positive integer): ")
        self.ngram_max_label.grid(row = 1, column = 2)
        
        self.ngram_max_text = tk.Entry(self.values_to_input)
        self.ngram_max_text.grid(row = 1, column = 3)
        
        self.min_df_label = tk.Label(self.values_to_input, text = "min doc freq. (float): ")
        self.min_df_label.grid(row = 2, column = 0)
        
        self.min_df_text = tk.Entry(self.values_to_input)
        self.min_df_text.grid(row = 2, column = 1)
        
        self.max_df_label = tk.Label(self.values_to_input, text = "max doc freq. (float): ")
        self.max_df_label.grid(row = 2, column = 2)
        
        self.max_df_text = tk.Entry(self.values_to_input)
        self.max_df_text.grid(row = 2, column = 3)
        
        self.topic_num_label = tk.Label(self.values_to_input, text = "Number of topics (positive integer): ")
        self.topic_num_label.grid(row = 3, column = 0)
        
        self.topic_num_text = tk.Entry(self.values_to_input)
        self.topic_num_text.grid(row = 3, column = 1)
        
        self.max_iter_label = tk.Label(self.values_to_input, text = "Max iteration (positive integer): ")
        self.max_iter_label.grid(row = 3, column = 2)
        
        self.max_iter_text = tk.Entry(self.values_to_input)
        self.max_iter_text.grid(row = 3, column = 3)
        
        
        '''
        #######################################
        ###Training data generation button
        #######################################
        '''
        
#        self.button_for_training = tk.Frame(self.button_groups)
#        self.button_for_training.pack(side = tk.LEFT)        
#        
#        self.user_drop_down_select_folder_buttom = tk.Button(self.button_for_training , text = "Select Folder\n(Training)")
#        self.user_drop_down_select_folder_buttom.pack()
#        self.user_drop_down_select_folder_buttom['command'] = self.select_folder_and_extract_xml
#        
#        self.user_drop_down_select_folder_button_create_vector = tk.Button(self.button_for_training ,
#                                                                           text = "Create Feature\nVector(s) (Training)")
#        self.user_drop_down_select_folder_button_create_vector.pack()
#        self.user_drop_down_select_folder_button_create_vector['command'] = self.create_feature_vector
#        
#        self.user_drop_down_select_folder_button_create_concept_prob = tk.Button(self.button_for_training,
#                                                                           text = "Create Concept\nProb(s) (Training)")
#        self.user_drop_down_select_folder_button_create_concept_prob.pack()
#        self.user_drop_down_select_folder_button_create_concept_prob['command'] = self.create_concept_matrix
#        
#        self.exit_button = tk.Button(self.root, text = 'quit')
        
        '''
        #######################################
        ###Testing data generation button
        #######################################
        '''
        
#        self.button_for_test = tk.Frame(self.button_groups)
#        self.button_for_test.pack(side = tk.LEFT)        
#        
#        self.user_drop_down_select_folder_buttom_test = tk.Button(self.button_for_test , text = "Select Folder\n(Tesing)")
#        self.user_drop_down_select_folder_buttom_test.pack()
#        self.user_drop_down_select_folder_buttom_test['command'] = self.select_folder_and_extract_xml_test
#        
#        self.user_drop_down_select_folder_button_create_vector_test = tk.Button(self.button_for_test ,
#                                                                           text = "Create Feature\nVector(s) (Tesing)")
#        self.user_drop_down_select_folder_button_create_vector_test.pack()
#        self.user_drop_down_select_folder_button_create_vector_test['command'] = self.create_feature_vector_test
#        
#        self.user_drop_down_select_folder_button_create_concept_prob_test = tk.Button(self.button_for_test,
#                                                                           text = "Create Concept\nProb(s) (Tesing)")
#        self.user_drop_down_select_folder_button_create_concept_prob_test.pack()
#        self.user_drop_down_select_folder_button_create_concept_prob_test['command'] = self.create_concept_matrix_test
#        
        '''
        #######################################
        ####Show the list of pre-existed CLDA model
        #######################################
        '''
#        self.CLDA_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
#        self.CLDA_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
#        
#        self.drop_down_CLDA_label_test = tk.Label(self.CLDA_generation_list_frame_test,
#                                                         text = "CLDA\nCreated List (Test)")
#        self.drop_down_CLDA_label_test.pack()
#        
#        self.drop_down_CLDA_frame_test = tk.Frame(self.CLDA_generation_list_frame_test)
#        self.drop_down_CLDA_frame_test.pack()
#        
#        self.drop_down_CLDA_list_test = tk.Listbox(self.drop_down_CLDA_frame_test,
#                                                          exportselection = 0)
#        
#        self.drop_down_CLDA_list_test.pack(side = tk.LEFT, fill = 'y')
#        
#        self.drop_down_CLDA_bar_test = tk.Scrollbar(self.drop_down_CLDA_frame_test, orient = "vertical")
#        self.drop_down_CLDA_bar_test.pack(side = tk.RIGHT, fill = 'y')
#        
#        self.drop_down_CLDA_list_test['yscrollcommand'] = \
#        self.drop_down_CLDA_bar_test.set
#        
#        self.drop_down_CLDA_bar_test['command'] = \
#        self.drop_down_CLDA_list_test.yview
#        
        '''
        #########################################
        ####Show the list of pre-existed LDA model
        #########################################
        '''
#        self.LDA_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
#        self.LDA_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
#        
#        self.drop_down_LDA_label_test = tk.Label(self.LDA_generation_list_frame_test,
#                                                         text = "LDA\nCreated List (Test)")
#        self.drop_down_LDA_label_test.pack()
#        
#        self.drop_down_LDA_frame_test = tk.Frame(self.LDA_generation_list_frame_test)
#        self.drop_down_LDA_frame_test.pack()
#        
#        self.drop_down_LDA_list_test = tk.Listbox(self.drop_down_LDA_frame_test,
#                                                          exportselection = 0)
#        
#        self.drop_down_LDA_list_test.pack(side = tk.LEFT, fill = 'y')
#        
#        self.drop_down_LDA_bar_test = tk.Scrollbar(self.drop_down_LDA_frame_test, orient = "vertical")
#        self.drop_down_LDA_bar_test.pack(side = tk.RIGHT, fill = 'y')
#        
#        self.drop_down_LDA_list_test['yscrollcommand'] = \
#        self.drop_down_LDA_bar_test.set
#        
#        self.drop_down_LDA_bar_test['command'] = \
#        self.drop_down_LDA_list_test.yview
        
        
        
        '''
        ##########################################
        ###Buttons
        ##########################################
        '''
#       
        
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack()
#        self.func_test = tk.Button(self.buttons_frame, text = 'Function test')
#        
#        self.func_test.pack(side = tk.BOTTOM)
#        self.func_test['command'] = self.test_all
        
        
        self.training_button = tk.Button(self.buttons_frame, text = 'training_data_creation_xml')
        
        self.training_button.grid(row = 0, column = 0)
        self.training_button['command'] = partial(self.asynchronous_topic_concept_retrieval,
                            self.select_folder_and_extract_xml_async, '.xml', dataset_dir, 0)

        
        self.training_button_txt = tk.Button(self.buttons_frame, text = 'training_data_creation_text')
        
        self.training_button_txt.grid(row = 0, column = 1)
        self.training_button_txt['command'] = partial(self.asynchronous_topic_concept_retrieval,
                                self.select_folder_and_extract_txt_async, '.txt', dataset_dir, 0)
        
        
        
        self.test_button = tk.Button(self.buttons_frame, text = 'test_data_creation_txt')
        self.test_button.grid(row = 1, column = 0)
        self.test_button['command'] =partial(self.asynchronous_data_retrieval_test, 
                        self.select_folder_and_extract_txt_async_test, '.txt', dataset_test)
        
        self.training_button_txt_test = tk.Button(self.buttons_frame, text = 'test_data_creation_xml')
        self.training_button_txt_test.grid(row = 1, column = 1)
        self.training_button_txt_test['command'] = partial(self.asynchronous_data_retrieval_test, 
                                     self.select_folder_and_extract_xml_async_test, '.xml', dataset_test)
        
        
#        
        self.LDA_button = tk.Button(self.buttons_frame, text = 'LDA_model_creation (training)')
        
        self.LDA_button.grid(row = 2, column = 0)
        self.LDA_button['command'] = partial(self.asynchronous_LDA_model_generation, dataset_dir, 0)
        
        self.CLDA_button = tk.Button(self.buttons_frame, text = 'CLDA_model_creation (training)')
        
        self.CLDA_button.grid(row = 2, column = 1)
        self.CLDA_button['command'] = partial(self.asynchronous_CLDA_model_generation, dataset_dir, 0)
        
        

        
        self.LDA_evaluation_screen = tk.Button(self.buttons_frame, text = 'Go to CLDA evaluation')
        
        self.LDA_evaluation_screen.grid(row = 3,columnspan = 3)
        self.LDA_evaluation_screen['command'] = self.move_to_CLDA_evaluation
        
        
        
        
        '''
        #######################################
        ####End the main menu
        #######################################
        '''
    #######################################################
    ########## Value input part
    #######################################################
    def retrieve_top_concept(self):
        
        try:
            if(self.top_concept_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: K = {}".format(default_K))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_K
            else:
                user_input_val = int(self.top_concept_text.get())
                if(user_input_val < 1):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nInput positive value!".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return 
                else:
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input top topic limit value {} is used".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input topic concept top limit value is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
    
    def retrieve_ngram_min(self):
        try:
            if(self.ngram_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: ngram = {}".format(default_ngram))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_ngram
            else:
                user_input_val = int(self.ngram_text.get())
                if(user_input_val < 1):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nInput positive value!".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return 
                else:
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input ngram value {} is used".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input ngram value is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
    
    def retrieve_ngram_max(self):
        try:
            if(self.ngram_max_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: ngram_max = {}".format(default_ngram))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_ngram
            else:
                user_input_val = int(self.ngram_max_text.get())
                if(user_input_val < 1):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nInput positive value!".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return 
                else:
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input ngram value {} is used".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input ngram value is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
    
    def retrieve_smooth_value(self):
        try:
            if(self.smooth_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: smooth = {}".format(default_smooth))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_smooth
            else:
                user_input_val = float(self.smooth_text.get())
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input ngram value {} is used".format(user_input_val))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input smooth is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
    
    def retrieve_max_df(self):
        try:
            if(self.max_df_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: max doc. freq. = {}".format(default_max_df))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_max_df
            else:
                user_input_val = float(self.max_df_text.get())
                if (user_input_val > 1.0 or user_input_val < 0.0):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nPlease input value between 0 < x < 1! (max doc. freq. {})".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input max doc. freq. {} is used".format(user_input_val))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input max doc. freq. is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
    
    def retrieve_min_df(self):
        try:
            if(self.min_df_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: min doc. freq. = {}".format(default_min_df))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_min_df
            else:
                user_input_val = float(self.min_df_text.get())
                if (user_input_val > 1.0 or user_input_val < 0.0):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nPlease input value between 0 < x < 1! (min doc. freq. {})".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input min doc. freq. value {} is used".format(user_input_val))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input min doc. freq. is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
   
    def retrieve_topic_num(self):
        try:
            if(self.topic_num_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: topic num. = {}".format(default_topic_num))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_topic_num
            else:
                user_input_val = int(self.topic_num_text.get())
                if(user_input_val < 2):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nInput positive value!".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return 
                else:
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input topic num. value {} is used".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input topic num. is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return 
    def retrieve_max_iter(self):
        try:
            if(self.max_iter_text.get() == ""):
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe default value is used: max iter. = {}".format(default_max_iter))
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                return default_max_iter
            else:
                user_input_val = int(self.max_iter_text.get())
                if(user_input_val < 1):
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nInput positive value!".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return 
                else:
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                    self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nThe input max iter. value {} is used".format(user_input_val))
                    self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: The input max iter. is invalid.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')

            return
#    Testing the function of all functions
    def test_all(self):
        top_concept_limit = self.retrieve_top_concept()
        if (top_concept_limit == None):
            return
        print(top_concept_limit)
        ngram_num = self.retrieve_ngram_min()
        if (ngram_num == None):
            return
        print(ngram_num)
        
        
        smooth_value = self.retrieve_smooth_value()
        if (smooth_value == None):
            return
        print(smooth_value)
        max_df_value = self.retrieve_max_df()
        if (max_df_value == None):
            return
        print(max_df_value)
        min_df_value = self.retrieve_min_df()
        if (min_df_value == None):
            return
        print(min_df_value)
        
        if(max_df_value <= min_df_value):
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: max_df <= min_df is not accepted")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            print("Error")
            return
        
        topic_num = self.retrieve_topic_num()
        if(topic_num == None):
            return
            
        print(topic_num)
        
        
        max_iter = self.retrieve_max_iter()
        if(max_iter == None):
            return
        print(max_iter)
        

        
        
        
    def move_to_CLDA_evaluation(self):
        
        
        self.root.destroy()
        
        # Destroy all the values in the self object
        del self
        
        # Conduct the garbage collection for reducing the memory usage...
        gc.collect()
        
        CLDA_eval_screen.main()
        
    
        
        
    def retrieve_topic_feature_concept_list(self):
        files_tmp = []
        files_tmp_test = []
        for dirpath, dirs, files in os.walk(dataset_dir):
            files_tmp.extend(files)
#        print(files_tmp)
            # only retrieve the files_tmp which end with .csv
            # Initialise the topic list
        self.topic_list = [x for x in files_tmp if x.endswith(file_name_df_suffix_csv)]
        for i in self.topic_list:
            self.user_drop_down_select_folder_list.insert(tk.END, i)
        # Initialise the features_list
        # Extract feature files from the file lists
        # No need to sort the values as the files are already sorted by names
        self.feature_list = [x for x in files_tmp if x.endswith(feature_matrix_suffix_csv)]
        
        for i in self.feature_list:
            self.drop_down_list_word_vector_list.insert(tk.END, i)
        
        self.concept_list = [x for x in files_tmp if x.endswith(concept_prob_suffix_json)]
        
        for i in self.concept_list:
            self.drop_down_concept_prob_vector_list.insert(tk.END, i)
        
           
        self.CLDA_list = [x for x in files_tmp if x.endswith(CLDA_suffix_pickle)]
        for i in self.CLDA_list:
            self.drop_down_CLDA_list.insert(tk.END, i)
            
        
        self.LDA_list = [x for x in files_tmp if x.endswith('_LDA.pkl')]
        for i in self.LDA_list:
            self.drop_down_LDA_list.insert(tk.END, i)
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_tmp_test.extend(files)

            # only retrieve the files_tmp_test which end with .csv
            # Initialise the topic list
        self.topic_list_test = [x for x in files_tmp_test if x.endswith(file_name_df_suffix_csv)]
        
        for i in self.topic_list_test:
            self.user_drop_down_select_folder_list_test.insert(tk.END, i)
            

            
            
    
    def select_folder_and_extract_xml_async(self,ask_folder, topic_list, dataset_dir):
  
        folder_directory = ask_folder
        

        temp_substr = os.path.basename(folder_directory)
            
        # If the processed file has already exists, then the process of the
        # topics will stop.
        if any(temp_substr in string for string in topic_list):
            print("topic already exist.")
            return 
        
        for_test_purpose_data = pd.DataFrame([], columns=['File','Topic', 'Text'])
        
        training_path = []
        for dirpath, dirs, files in os.walk(folder_directory):
            training_path.extend(files)
    
        # Remove the files other than xml files
        training_path = [x for x in training_path if x.endswith('xml')]
        print(training_path)
        topic_name = os.path.basename(folder_directory)
        
        # Create the folder
        # if there is no directory storing the generated txt files
        if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
                
        conv_folder = dataset_dir + '/' + converted_folder
        if not os.path.isdir(conv_folder):
            os.makedirs(conv_folder)
            
        topic_folder = conv_folder + '/' + topic_name 
        if not os.path.isdir(topic_folder):
            os.makedirs(topic_folder)
        
        for path_to_file in training_path:
            path_string = os.path.basename(os.path.normpath(path_to_file)) 
            
            # Extract xml information from the files
            # then it construct the information
            file = folder_directory + '/' + path_string 
            tree = ET.parse(file)
            root = tree.getroot()
            result = ''
            for element in root.iter():
                if(element.text != None):
                    result += element.text + ' '
            # Remove the remained data
            result = result[:-1]
            
            name_of_the_file = (os.path.basename(path_string))
            
            for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(name_of_the_file, 
                                                                               topic_name,
                                                                               result)], 
            columns=['File','Topic', 'Text']))

            # Write the xml results as files    
            with open(topic_folder + '/' + path_to_file[:-len('.xml')] + converted_xml_suffix, "w") as f:
                f.write(result)
                
                
            
            
        if(len(for_test_purpose_data) != 0):
            for_test_purpose_data.to_csv(dataset_dir + '/' +
                              topic_name + file_name_df_suffix_csv,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + file_name_df_suffix_csv
    
    
    # Select the labelling folder to label the dataset....
#    def select_labelling_folder():
#        
#        
#        return

    def select_folder_and_extract_txt_async(self,ask_folder, topic_list, dataset_dir):
  
        folder_directory = ask_folder
        

        temp_substr = os.path.basename(folder_directory)
            
        # If the processed file has already exists, then the process of the
        # topics will stop.
        if any(temp_substr in string for string in topic_list):
            print("topic already exist.")
            return 
        
        for_test_purpose_data = pd.DataFrame([], columns=['File','Topic', 'Text'])
        
        training_path = []
        for dirpath, dirs, files in os.walk(folder_directory):
            training_path.extend(files)
            
        # Remove the files other than xml files
        training_path = [x for x in training_path if x.endswith('.txt') and not x.startswith('._')]
#        print(training_path)
        topic_name = os.path.basename(folder_directory)
       
        for path_to_file in training_path:
            path_string = os.path.basename(os.path.normpath(path_to_file)) 
            
            file = folder_directory + '/' + path_string
            f = open(file, "r")
            result = f.read()
            
            name_of_the_file = (os.path.basename(path_string))
            
            for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(name_of_the_file, 
                                                                               topic_name,
                                                                               result)], 
            columns=['File','Topic', 'Text']))
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
        if(len(for_test_purpose_data) != 0):
            for_test_purpose_data.to_csv(dataset_dir + '/' +
                              topic_name + file_name_df_suffix_csv,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + file_name_df_suffix_csv
        
        
    def select_folder_and_extract_txt_async_test(self,ask_folder, topic_list, dataset_dir, label_dir):
  
        folder_directory = ask_folder
        
#        print(folder_directory)
        temp_substr = os.path.basename(folder_directory)
            
        # If the processed file has already exists, then the process of the
        # topics will stop.
        if any(temp_substr in string for string in topic_list):
            print("topic already exist.")
            return 
        
        for_test_purpose_data = pd.DataFrame([], columns=['File','Topic', 'Text'])
        
        training_path = []
        for dirpath, dirs, files in os.walk(folder_directory):
            training_path.extend(files)
        
        # Remove the files other than xml files
        training_path = [x for x in training_path if x.endswith('.txt') and not x.startswith('._')]
        
        training_path = sorted(training_path)
        topic_name = os.path.basename(folder_directory)
        
        name_of_the_label_text = os.path.basename(folder_directory) + label_text_suffix
        
        label_data = pd.read_csv(label_dir + '/' + name_of_the_label_text, 
                        encoding='utf-8',
                        delimiter = label_delim,
                        quoting=csv.QUOTE_ALL, names = [''.join(filter(str.isdigit, name_of_the_label_text)), 'File', 'label'])
        
        label_data = label_data.sort_values(by = ['File'])
#        dataframe_ = pd.DataFrame([], columns=['File','Topic', 'Text'])
        labelling = list(label_data['label'])
#        labelling = list(label_data['0'])
        for_test_purpose_data = []
        for path_to_file in training_path:
            path_string = os.path.basename(os.path.normpath(path_to_file)) 
            
            file = folder_directory + '/' + path_string
            f = open(file, "r")
            result = f.read()
            
            name_of_the_file = (os.path.basename(path_string))
            
            for_test_purpose_data.append((name_of_the_file, topic_name,result))
            
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
                
                
        a,b,c = zip(*for_test_purpose_data)
        
        framing = [(x,y,z,w) for x,y,z,w in zip(a,b,c,labelling)]
        dataframe_ = pd.DataFrame(framing, columns=['File','Topic', 'Text','label'])        
            
        if(len(dataframe_) != 0):
            dataframe_.to_csv(dataset_dir + '/' +
                              topic_name + file_name_df_suffix_csv,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + file_name_df_suffix_csv
    
    
    
    def select_folder_and_extract_xml_async_test(self,ask_folder, topic_list, dataset_dir, label_dir):
        
        folder_directory = ask_folder

        temp_substr = os.path.basename(folder_directory)
            
        # If the processed file has already exists, then the process of the
        # topics will stop.
        if any(temp_substr in string for string in topic_list):
            print("topic already exist.")
            return 
        
        
            
        training_path = []
        for dirpath, dirs, files in os.walk(folder_directory):
            training_path.extend(files)
        
        # Remove the files other than xml files
        training_path = [x for x in training_path if x.endswith('xml')]
#        print(training_path)
        topic_name = os.path.basename(folder_directory)
        
        # Create the folder
        # if there is no directory storing the generated txt files
        if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
                
        conv_folder = dataset_dir + '/' + converted_folder
        if not os.path.isdir(conv_folder):
            os.makedirs(conv_folder)
            
        topic_folder = conv_folder + '/' + topic_name 
        if not os.path.isdir(topic_folder):
            os.makedirs(topic_folder)
        # Labelled data is automatically sought based on the 
        # collection of data
        training_path = sorted(training_path)
        
        name_of_the_label_text = os.path.basename(folder_directory) + label_text_suffix
        # print(path_string)
        
        label_data = pd.read_csv(label_dir + '/' + name_of_the_label_text, 
                        encoding='utf-8',
                        delimiter = label_delim,
                        quoting=csv.QUOTE_ALL, names = [''.join(filter(str.isdigit, name_of_the_label_text)), 'File', 'label'])
        
        
        label_data = label_data.sort_values(by = ['File'])
#        dataframe_ = pd.DataFrame([], columns=['File','Topic', 'Text'])
        labelling = list(label_data['label'])
#        labelling = list(label_data['0'])
        for_test_purpose_data = []
        for path_to_file in training_path:
            path_string = os.path.basename(os.path.normpath(path_to_file)) 
            
            # Extract xml information from the files
            # then it construct the information
            file = folder_directory + '/' + path_string 
            tree = ET.parse(file)
            root = tree.getroot()
            result = ''
            for element in root.iter():
                if(element.text != None):
                    result += element.text + ' '
            # Remove the remained data
            result = result[:-1]
            
            name_of_the_file = (os.path.basename(path_string))            
            
            for_test_purpose_data.append((name_of_the_file, topic_name,result))
            
            
            # Write the xml results as files    
            with open(topic_folder + '/' + path_to_file[:-len('.xml')] + converted_xml_suffix, "w") as f:
                f.write(result)
        
        # for_test_purpose_data.loc[:, 'label'] = label_data['label']
        a,b,c = zip(*for_test_purpose_data)
        
        framing = [(x,y,z,w) for x,y,z,w in zip(a,b,c,labelling)]
        dataframe_ = pd.DataFrame(framing, columns=['File','Topic', 'Text','label'])
        
        
        if(len(dataframe_) != 0):
            dataframe_.to_csv(dataset_dir + '/' +
                              topic_name + file_name_df_suffix_csv,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + file_name_df_suffix_csv
    
    
    def asynchronous_data_retrieval_test(self, fobj, file_type, dataset_dir):
        sys.stdout = buffer = StringIO()
        # Select the testing folder
        train_folder_selection = askdirectory()
        
        label_folder_selection = askdirectory()
        
        training_folders_tmp = []
        for dirpath, dirs, files in os.walk(train_folder_selection):
            if len([x for x in files if x.endswith(file_type)]) != 0:
                training_folders_tmp.append(dirpath)
        
#        print(training_folders_tmp)
        async def retrieve_file_data(training_folders_tmp, topic_list):
        # Max worker set to 10
           with concurrent.futures.ThreadPoolExecutor() as executor:
            
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
#                        self.select_folder_and_extract_xml_async
                        fobj, 
                        folder_name,
                        topic_list, dataset_dir, label_folder_selection
                    )
                    for folder_name in training_folders_tmp
                ]
                topics_vec = []
                for i in await asyncio.gather(*futures):
                    topics_vec.append(i)
                    
                return topics_vec
        
        topic_list = self.topic_list_test    
        
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(retrieve_file_data(training_folders_tmp, topic_list))
        topics = loop.run_until_complete(future)  
        
        if type(topics) == list:
            self.topic_list_test.extend(topics)
            self.topic_list_test = sorted(self.topic_list)
            
        if type(topics) == list:
            for i in topics:
                    self.user_drop_down_select_folder_list_test.insert(tk.END, i)
        print("Test data retrieval completed!")
        
        sys.stdout = sys.__stdout__
        self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
        self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, buffer.getvalue())
        self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
    # Labelling test folder
        
    # Retrieving topic list
    
    # Based on the files_tmp made by the vectors 
    # Depending on what test vector u used 
    # The contents of the vector can be changed 


    # Retrieve all test and training data asynchrously
    
    def asynchronous_topic_concept_retrieval(self, fobj, file_type, dataset_dir, test_or_training):
        sys.stdout = buffer = StringIO()
        
        train_folder_selection = askdirectory()
        #train_folder_selection = "C:/Users/n9648852/Desktop/R8-Dataset/Dataset/R8/Testing"
        
        if(train_folder_selection == ""):
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nFolder not selected. Abort.")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            return
        
        training_folders_tmp = []
        
        for dirpath, dirs, files in os.walk(train_folder_selection):
            if len([x for x in files if x.endswith(file_type)]) != 0:
                training_folders_tmp.append(dirpath) 
        
        top_concept_limit = self.retrieve_top_concept()
        if (top_concept_limit == None):
            return
        
        ngram_num_min = self.retrieve_ngram_min()
        if (ngram_num_min == None):
            return
        
        ngram_num_max = self.retrieve_ngram_max()
        if (ngram_num_max == None):
            return
        
        smooth_value = self.retrieve_smooth_value()
        if (smooth_value == None):
            return
        
        max_df_value = self.retrieve_max_df()
        if (max_df_value == None):
            return
        
        min_df_value = self.retrieve_min_df()
        if (min_df_value == None):
            return
        
        
        if(max_df_value <= min_df_value):
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: max_df <= min_df is not accepted")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            print("Error")
            return
        
        if(ngram_num_max <  ngram_num_min):
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, "\nError: max_ngram < min_ngram is not accepted")
            self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
            print("Error")
            return
        
        
        ngram_list = (ngram_num_min, ngram_num_max)
        # training_folders_tmp.remove(train_folder_selection)
        
        if(test_or_training == 0):
            topic_list = self.topic_list
        else:
            topic_list = self.topic_list_test
        '''
        Asynchronous training dataset retrieval
        '''
        # Retrieve file (either text or xml) data and then
        # Stores them as one csv file
        # Each row corresponds to the file
        # which this method have retrieved from the folder
        async def retrieve_file_data(training_folders_tmp, topic_list):
        # Max worker set to 10
           with concurrent.futures.ThreadPoolExecutor() as executor:
            
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
#                        self.select_folder_and_extract_xml_async
                        fobj, 
                        folder_name,
                        topic_list, dataset_dir
                    )
                    for folder_name in training_folders_tmp
                ]
                topics_vec = []
                for i in await asyncio.gather(*futures):
                    topics_vec.append(i)
                    
                return topics_vec 
            
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(retrieve_file_data(training_folders_tmp, topic_list))
        topics = loop.run_until_complete(future)
        print("Data extraction completed!")
#        print(topics)
        if None in topics:
            topics = list(filter((None).__ne__, topics))
#        print(topics)
        if (test_or_training == 0):
            if type(topics) == list:
                self.topic_list.extend(topics)
            self.topic_list = sorted(self.topic_list)
            
            if type(topics) == list:
                for i in topics:
                    self.user_drop_down_select_folder_list.insert(tk.END, i)
        else:
            if type(topics) == list:
                self.topic_list_test.extend(topics)
            self.topic_list_test = sorted(self.topic_list)
            
            if type(topics) == list:
                for i in topics:
                    self.user_drop_down_select_folder_list_test.insert(tk.END, i)
        
        
        
        time.sleep(4)
        
        
        if(test_or_training == 0):
            feature_list = self.feature_list
        else:
            feature_list = self.feature_list_test
        # As nltk is not thread safe model generator
        # this method is applied to create massive amount of 
        # LDA model later on
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(create_feature_vector_async, [[i, feature_list, dataset_dir, 
                                                                        ngram_list,  min_df_value,
                                                                        max_df_value] for i in training_folders_tmp])
            features = pool_async.get()
        
#        loop = asyncio.get_event_loop()
#        future = asyncio.ensure_future(create_feature_vectors(training_folders_tmp, feature_list, dataset_dir))
#        features = loop.run_until_complete(future)
        
        
        if None in features:
            features = list(filter((None).__ne__, features))
            
        if(test_or_training == 0):
            if type(features) == list:    
                self.feature_list.extend(features)
                
            self.feature_list = sorted(self.feature_list)
            
            if type(features) == list:
                for i in features:
                    self.drop_down_list_word_vector_list.insert(tk.END, i)
        else:
            if type(features) == list:    
                self.feature_list_test.extend(features)
                
            self.feature_list_test = sorted(self.feature_list_test)
            
            if type(features) == list:
                for i in features:
                    self.drop_down_list_word_vector_list_test.insert(tk.END, i)
        print("Feature extraction completed!!")
        
        time.sleep(2)
        sys.stdout = sys.__stdout__
        
        # Asynchrously retrieve the probability p(word|concept) from
        # Probase
        def create_concept_word_lists(training_folders_tmp, concept_list, dataset_dir):
            concept_vec = [] #Move out just in case...
#            for i in training_folders_tmp:
#                concept_vec.append(create_concept_matrix_async(i, concept_list, dataset_dir))
            
            with Pool(cpu_count()-1) as p:
                pool_async = p.starmap_async(create_concept_matrix_async, [[i, concept_list, dataset_dir,top_concept_limit,smooth_value] for i in training_folders_tmp])
                concept_vec = pool_async.get()
            time.sleep(2)    #Sleep just in case...
            return concept_vec
        
        if(test_or_training == 0):
            concept_list = self.concept_list
        else:
            concept_list = self.concept_list_test


        concepts = create_concept_word_lists(training_folders_tmp, concept_list, dataset_dir)
        
        if None in concepts:
            concepts = list(filter((None).__ne__, concepts))
        
        
        if(test_or_training == 0):
            if type(concepts) == list:    
                self.concept_list.extend(concepts)
            
            
            self.concept_list = sorted(self.concept_list)
            
            
            if type(concepts) == list:
                for i in concepts:
                    self.drop_down_concept_prob_vector_list.insert(tk.END, i)
        else:
            if type(concepts) == list:    
                self.concept_list_test.extend(concepts)
            
            
            self.concept_list_test = sorted(self.concept_list_test)
            
            
            if type(concepts) == list:
                for i in concepts:
                    self.drop_down_concept_prob_vector_list_test.insert(tk.END, i)
        print("Concept graph retrieval completed!!")
        self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
        self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, buffer.getvalue())
        self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')

        
    def asynchronous_CLDA_model_generation(self, dataset_dir, result_num):
        
        topic_num = self.retrieve_topic_num()
        if(topic_num == None):
            return
            
        max_iter = self.retrieve_max_iter()
        if(max_iter == None):
            return
        
        def concurrent():
            
            fm = Asynchrous_CLDA()
            
            results = fm.asynchronous_CLDA_creation(dataset_dir, topic_num, max_iter)
            
            return results
        
        results = concurrent()
        # Create CLDA object asynchronically.
        
        for i in results:
            line = i[1].getvalue()
            if line != "":
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, line)
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
        # If result num is equal to 1, then the result can be put into 1
        
        if(result_num == 0):
            for i in results:
                if(i[0] != None):
                    self.drop_down_CLDA_list.insert(tk.END, i[0])
        # Otherwise, the value is put into the values
        else:
            for i in results:
                if(i[0] != None):
                    self.drop_down_CLDA_list_test.insert(tk.END, i[0])
        
        # Make the stdout setting to default just in case...
        sys.stdout = sys.__stdout__            

    def asynchronous_LDA_model_generation(self, dataset_dir, result_num):
        
        topic_num = self.retrieve_topic_num()
        if(topic_num == None):
            return

        max_iter = self.retrieve_max_iter()
        if(max_iter == None):
            return
        
        def concurrent():
        #        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
            
            fm = Asynchrous_LDA
            
            results = fm.asynchronous_LDA_creation(fm, dataset_dir, topic_num, max_iter)
            
            return results
            
        print("LDA model creation start!")
        results = concurrent()
        # Create CLDA object asynchronously
        
        for i in results:
            line = i[1].getvalue()
            if line != "":
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='normal')
                self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, line)
                self.user_drop_down_folder_selection_results_scroll_list.configure(state='disabled')
        # If result num is equal to 1, then the result can be put into 1
        
        if(result_num == 0):
            for i in results:
                if(i[0] != None):
                    self.drop_down_LDA_list.insert(tk.END, i[0])
        # Otherwise, the value is put into the values
        else:
            for i in results:
                if(i[0] != None):
                    self.drop_down_LDA_list_test.insert(tk.END, i[0])
        
        # Make the stdout setting to default just in case...
        sys.stdout = sys.__stdout__
        
            
#Asynchronous CLDA model creation            
class Asynchrous_CLDA(object):
    
    def __init__(self):
#        self.__init__()
        self.dataset_dir = None
        
        
    def create_CLDA_instance(self,i, topic_num, max_iter):
        # turn standard out into string IO...
        # to capture the standard output
        sys.stdout = buffer = StringIO()
        # Read CLDA files 
        files_tmp = []
        for dirpath, dirs, files in os.walk(self.dataset_dir):
            if len([x for x in files if x.endswith(CLDA_suffix_pickle)]) != 0:
                files_tmp.extend(files) 
        
        # 
        if i + CLDA_suffix_pickle in files_tmp:
            print("File {} is already exists".format(i + CLDA_suffix_pickle))
            # Normal finish of the program...
            return None, ""
        
        print("file process {}: starts!".format(i))
        file_index_name = pd.read_csv(self.dataset_dir + '/' + i + file_name_df_suffix_csv, encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)["File"]

            
            
        feature_matrix, feature_names = (None, [])
        
        # Load the feature_name from a txt file
        with open(self.dataset_dir + '/' + i + feature_name_suffix_txt, "r") as f: 
            for line in f:
                #Remove the \n
                feature_names.append(line.strip('\n'))
        
        # Load the matrix data from a csv file
        with open(self.dataset_dir + '/' + i + feature_matrix_suffix_csv, "r") as f:
            feature_matrix = np.loadtxt(f, delimiter = delim)
        
        concept_dict, concept_names = (None, [])
        
        # Load the word-concept probability data from a json file
        with open(self.dataset_dir + '/' + i + concept_prob_suffix_json, "r") as f:
            concept_dict = json.load(f)
            
        # Load concept name data from a text file
        with open(self.dataset_dir + '/' + i + concept_name_suffix_txt, "r") as f:
            for line in f:
                concept_names.append(line.strip('\n'))
            
            
        print("file reading {}: complete!".format(i))
        
        # Forcefully create standard output
        sys.stdout.flush()
        
        # Create CLDA instance
        CLDA_instance = CLDA.CLDA(feature_names, concept_names, file_index_name,topic_num, max_iter)
        
        # Run CLDA
        CLDA_instance.run(feature_matrix, concept_dict)
        
        # The created CLDA object is stored in the files
        with open(self.dataset_dir + '/' + i + CLDA_suffix_pickle, "wb") as f:
            pickle.dump(CLDA_instance, f)
        
        # Sleep just in case...
        time.sleep(0.5)
        sys.stdout = sys.__stdout__
        
        # Return True if the process stops normally
        return (i + CLDA_suffix_pickle, buffer)
    
    def asynchronous_CLDA_creation(self, dataset_dir, topic_num, max_iter):
           
        self.dataset_dir = dataset_dir    
        files_tmp = []
        for dirpath, dirs, files in os.walk(self.dataset_dir):
            if len([x for x in files if x.endswith(file_name_df_suffix_csv)]) != 0:
                files_tmp.extend(files) 
        
        files_list_for_modelling_CLDA = sorted(list(set([x[:-len(file_name_df_suffix_csv)] for x in files_tmp if x.endswith(file_name_df_suffix_csv)])))
        
        # Asynchronically create the CLDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.create_CLDA_instance, [[i, topic_num, max_iter] for i in files_list_for_modelling_CLDA])
            
            # Return processed result
            return pool_async.get()
                
        
class Asynchrous_LDA(object):
    
    def __init__(self):
        self.__init__(self)
        self.dataset_dir = None
        
        
    def create_LDA_instance(self,i, dataset_dir, topic_num, max_iter):
        sys.stdout = buffer = StringIO()  
        files_tmp = []
        
        # Looking around the file with topic name i
        for dirpath, dirs, files in os.walk(dataset_dir):
            if len([x for x in files if x.endswith(LDA_suffix_pickle)]) != 0:
                files_tmp.extend(files) 
        print(dataset_dir + '/' + i + LDA_suffix_pickle)
        
        # If the file has already been created 
        # then skip the process...
        if i + LDA_suffix_pickle in files_tmp:
            print("File {} is already exists".format(i + LDA_suffix_pickle))
            #Normal finish of the program...
            return None, ""
        
        # Read CLDA files 
        print("file process {}: starts!".format(i))
        file_index_name = pd.read_csv(dataset_dir + '/' + i + file_name_df_suffix_csv, encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)["File"]
        
        feature_matrix, feature_names = (None, [])
        
        # Load the feature_name from a txt file
        with open(dataset_dir + '/' + i + feature_name_suffix_txt, "r") as f: 
            for line in f:
                #Remove the \n
                feature_names.append(line.strip('\n'))
        # Load the matrix data from a csv file
        with open(dataset_dir + '/' + i + feature_matrix_suffix_csv, "r") as f:
            feature_matrix = np.loadtxt(f, delimiter = delim)
        

        
        # Forcefully make standard output
        sys.stdout.flush()
        
        # Create LDA instance
        LDA_instance = CLDA.LDA(file_index_name, feature_names, topic_num)
        
        # Fit LDA model to dataset
        LDA_instance.run(feature_matrix, max_iter)
        
        # Save fitted LDA model
        with open(dataset_dir + '/' + i + LDA_suffix_pickle, "wb") as f:
            pickle.dump(LDA_instance, f)
        
        # Sleep just in case...
        print("Model generation complete!: {}".format(dataset_dir + '/' + i + LDA_suffix_pickle))
        time.sleep(0.5)
        
        # Return True if the process stops normally
        sys.stdout = sys.__stdout__
        return (i + LDA_suffix_pickle, buffer)
    
    def asynchronous_LDA_creation(self, dataset_dir, topic_num, max_iter):
           
          
        # Initialize file_tmp list
        files_tmp = []
        
        # Walk down the files to search for
        # files to geenrate model
        for dirpath, dirs, files in os.walk(dataset_dir):
            if len([x for x in files if x.endswith(file_name_df_suffix_csv)]) != 0:
                print(files)
                files_tmp.extend(files) 
        
        # Very rough file detection....
        files_list_for_modelling_LDA = sorted(list(set([x[:-len(file_name_df_suffix_csv)] for x in files_tmp if x.endswith(file_name_df_suffix_csv)])))
        
        # Core use
        # Asynchronically create the LDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.create_LDA_instance, [[self, i, dataset_dir, topic_num, max_iter] for i in files_list_for_modelling_LDA])
            
            # Return processed result
            return pool_async.get()        
            

        
def main():
    #Run the main GUI application
    Application()


if __name__ == "__main__":
    main()
