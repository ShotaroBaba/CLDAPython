# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:45:47 2018

@author: Shotaro Baba
"""
# Reading all necessary packages

# Import all necessary packages.import sys
import sys
sys.path.append("..\/models/")
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import CLDA_eval_screen
import CLDA
import gc

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

from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import Process


import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

import asyncio
import concurrent.futures
import requests

lemmatizer = WordNetLemmatizer()
dataset_dir = "../../CLDA_data_training"
dataset_test = "../../CLDA_data_testing"
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

def define_sw():
    
    return set(stopwords.words('english'))# + stop_words)

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
def generate_vector():
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=[1,2],
                           min_df=0.02, max_df=0.98)

# Generate count vectorizer
def vectorize(tf_vectorizer, df):
    # Generate_vector
    # df = df.reindex(columns=['text'])  # reindex on tweet

    tf_matrix = tf_vectorizer.fit_transform(df['Text'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    return tf_matrix, tf_feature_names


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
#    for_test_purpose_data = {}
#    training_data_list = []
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        
        # training_data_list.append(path_string)
        # Initialise the list of the strings
        # for_test_purpose_data[path_string] = {}
        
        file = test_path + '/' + path_string 
        tree = ET.parse(file)
        
        # Turn the string into
        root = tree.getroot()
        result = ''
        # Retrieving the text from xml files
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        result = result[:-1] #Removing the space
        
                
        # Initialise 
#        for file in generate_files(path):
            #print("Now reading...")
            #print(open(file, "r").read())

        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']))
    
    return for_test_purpose_data



 
class Application():
    
        
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("File Pre Processing & Model Creation")
#        self.pack()
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
       
        #The list of the files for the upload
        self.upload_file_list_labelling_test = []
        self.upload_file_list_classifier_test = []
        
        self.linked_folders_test = []
        self.folder_directory_test = None
        self.document_data_test = []
        
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
        ####Show the list of pre-existed CLDA model
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
        
        self.user_drop_down_select_folder_label_test = tk.Label(self.frame_for_folder_selection_test , text = "Folder (Topic)\nSelection")
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
        
        self.word_vector_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
        self.word_vector_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_list_word_vector_label_test = tk.Label(self.word_vector_generation_list_frame_test,
                                                         text = "Word Vector\nCreated List (Test)")
        self.drop_down_list_word_vector_label_test.pack()
        
        self.drop_down_list_word_vector_frame_test = tk.Frame(self.word_vector_generation_list_frame_test)
        self.drop_down_list_word_vector_frame_test.pack()
        
        self.drop_down_list_word_vector_list_test = tk.Listbox(self.drop_down_list_word_vector_frame_test,
                                                          exportselection = 0)
        
        self.drop_down_list_word_vector_list_test.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_list_word_vector_bar_test = tk.Scrollbar(self.drop_down_list_word_vector_frame_test, orient = "vertical")
        self.drop_down_list_word_vector_bar_test.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_list_word_vector_list_test['yscrollcommand'] = \
        self.drop_down_list_word_vector_bar_test.set
        
        self.drop_down_list_word_vector_bar_test['command'] = \
        self.drop_down_list_word_vector_list_test.yview
    
        
        '''
        #######################################
        ####Concpet-word vector extraction
        #######################################
        '''
        
        self.concept_prob_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
        self.concept_prob_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_concept_prob_vector_label_test = tk.Label(self.concept_prob_generation_list_frame_test,
                                                         text = "Concept Prob\nCreated List (Test)")
        self.drop_down_concept_prob_vector_label_test.pack()
        
        self.drop_down_concept_prob_vector_frame_test = tk.Frame(self.concept_prob_generation_list_frame_test)
        self.drop_down_concept_prob_vector_frame_test.pack()
        
        self.drop_down_concept_prob_vector_list_test = tk.Listbox(self.drop_down_concept_prob_vector_frame_test,
                                                          exportselection = 0)
        
        self.drop_down_concept_prob_vector_list_test.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_concept_prob_vector_bar_test = tk.Scrollbar(self.drop_down_concept_prob_vector_frame_test, orient = "vertical")
        self.drop_down_concept_prob_vector_bar_test.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_concept_prob_vector_list_test['yscrollcommand'] = \
        self.drop_down_concept_prob_vector_bar_test.set
        
        self.drop_down_concept_prob_vector_bar_test['command'] = \
        self.drop_down_concept_prob_vector_list_test.yview
        
        
        '''
        ################################################################################
        ###End of retrieving the test data part 
        ################################################################################
        '''
        
        # Forming the button group for putting them together
        self.button_groups = tk.Frame(self.main_listbox_and_buttons)
        self.button_groups.pack()
        
        
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
        self.CLDA_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
        self.CLDA_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_CLDA_label_test = tk.Label(self.CLDA_generation_list_frame_test,
                                                         text = "CLDA\nCreated List (Test)")
        self.drop_down_CLDA_label_test.pack()
        
        self.drop_down_CLDA_frame_test = tk.Frame(self.CLDA_generation_list_frame_test)
        self.drop_down_CLDA_frame_test.pack()
        
        self.drop_down_CLDA_list_test = tk.Listbox(self.drop_down_CLDA_frame_test,
                                                          exportselection = 0)
        
        self.drop_down_CLDA_list_test.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_CLDA_bar_test = tk.Scrollbar(self.drop_down_CLDA_frame_test, orient = "vertical")
        self.drop_down_CLDA_bar_test.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_CLDA_list_test['yscrollcommand'] = \
        self.drop_down_CLDA_bar_test.set
        
        self.drop_down_CLDA_bar_test['command'] = \
        self.drop_down_CLDA_list_test.yview
        
        '''
        #########################################
        ####Show the list of pre-existed LDA model
        #########################################
        '''
        self.LDA_generation_list_frame_test = tk.Frame(self.main_listbox_and_result_test)
        self.LDA_generation_list_frame_test.pack(side = tk.LEFT, anchor = tk.N)
        
        self.drop_down_LDA_label_test = tk.Label(self.LDA_generation_list_frame_test,
                                                         text = "LDA\nCreated List (Test)")
        self.drop_down_LDA_label_test.pack()
        
        self.drop_down_LDA_frame_test = tk.Frame(self.LDA_generation_list_frame_test)
        self.drop_down_LDA_frame_test.pack()
        
        self.drop_down_LDA_list_test = tk.Listbox(self.drop_down_LDA_frame_test,
                                                          exportselection = 0)
        
        self.drop_down_LDA_list_test.pack(side = tk.LEFT, fill = 'y')
        
        self.drop_down_LDA_bar_test = tk.Scrollbar(self.drop_down_LDA_frame_test, orient = "vertical")
        self.drop_down_LDA_bar_test.pack(side = tk.RIGHT, fill = 'y')
        
        self.drop_down_LDA_list_test['yscrollcommand'] = \
        self.drop_down_LDA_bar_test.set
        
        self.drop_down_LDA_bar_test['command'] = \
        self.drop_down_LDA_list_test.yview
        
        
        
        '''
        ##########################################
        ###Buttons
        ##########################################
        '''
        
        self.training_button = tk.Button(self.root, text = 'training_data_creation_xml')
        
        self.training_button.pack(side = tk.BOTTOM)
        self.training_button['command'] = partial(self.asynchronous_topic_concept_retrieval,
                            self.select_folder_and_extract_xml_async, '.xml', dataset_dir, 0)

        
        self.training_button_txt = tk.Button(self.root, text = 'training_data_creation_text')
        
        self.training_button_txt.pack(side = tk.BOTTOM)
        self.training_button_txt['command'] = partial(self.asynchronous_topic_concept_retrieval,
                                self.select_folder_and_extract_txt_async, '.txt', dataset_dir, 0)
        
        
        
        self.test_button = tk.Button(self.root, text = 'test_data_creation_txt')
        self.test_button.pack(side = tk.BOTTOM)
        self.test_button['command'] =partial(self.asynchronous_topic_concept_retrieval, 
                        self.select_folder_and_extract_txt_async, '.txt', dataset_test,1)
        
        self.training_button_txt_test = tk.Button(self.root, text = 'test_data_creation_xml')
        self.training_button_txt_test.pack(side = tk.BOTTOM)
        self.training_button_txt_test['command'] = partial(self.asynchronous_topic_concept_retrieval, 
                                     self.select_folder_and_extract_xml_async, '.xml', dataset_test, 1)
        
        
#        self.CLDA_button = tk.Button(self.root, text = 'CLDA_model_creation (training)')
#        
#        self.training_button_txt.pack(side = tk.BOTTOM)
#        self.training_button_txt['command'] = partial(self.asynchronous_topic_concept_retrieval, self.select_folder_and_extract_txt_async, '.txt')
#        
        self.LDA_button = tk.Button(self.root, text = 'LDA_model_creation (training)')
        
        self.LDA_button.pack(side = tk.BOTTOM)
        self.LDA_button['command'] = partial(self.asynchronous_LDA_model_generation, dataset_dir, 0)
        
        self.CLDA_button = tk.Button(self.root, text = 'CLDA_model_creation (training)')
        
        self.CLDA_button.pack(side = tk.BOTTOM)
        self.CLDA_button['command'] = partial(self.asynchronous_CLDA_model_generation, dataset_dir, 0)
        
        
        
        self.LDA_button_test = tk.Button(self.root, text = 'LDA_model_creation (test)')
        
        self.LDA_button_test.pack(side = tk.BOTTOM)
        self.LDA_button_test['command'] = partial(self.asynchronous_LDA_model_generation, dataset_test, 1)
        
        
        
        self.CLDA_button_test = tk.Button(self.root, text = 'CLDA_model_creation (test)')
        
        self.CLDA_button_test.pack(side = tk.BOTTOM)
        self.CLDA_button_test['command'] = partial(self.asynchronous_CLDA_model_generation, dataset_test, 1)
        
        
        self.LDA_evaluation_screen = tk.Button(self.root, text = 'Go to CLDA evaluation')
        
        self.LDA_evaluation_screen.pack(side = tk.BOTTOM)
        self.LDA_evaluation_screen['command'] = self.move_to_CLDA_evaluation
        
        
        
        
        '''
        #######################################
        ####End the main menu
        #######################################
        '''
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
        self.topic_list = [x for x in files_tmp if x.endswith('.csv')]
        for i in self.topic_list:
            self.user_drop_down_select_folder_list.insert(tk.END, i)
        # Initialise the features_list
        # Extract feature files from the file lists
        # No need to sort the values as the files are already sorted by names
        self.feature_list = [x for x in files_tmp if x.endswith('_f.pkl')]
        
        for i in self.feature_list:
            self.drop_down_list_word_vector_list.insert(tk.END, i)
        
        self.concept_list = [x for x in files_tmp if x.endswith('_c.pkl')]
        
        for i in self.concept_list:
            self.drop_down_concept_prob_vector_list.insert(tk.END, i)
        
           
        self.CLDA_list = [x for x in files_tmp if x.endswith('_CLDA.pkl')]
        for i in self.CLDA_list:
            self.drop_down_CLDA_list.insert(tk.END, i)
            
        
        self.LDA_list = [x for x in files_tmp if x.endswith('_LDA.pkl')]
        for i in self.LDA_list:
            self.drop_down_LDA_list.insert(tk.END, i)
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_tmp_test.extend(files)

            # only retrieve the files_tmp_test which end with .csv
            # Initialise the topic list
        self.topic_list_test = [x for x in files_tmp_test if x.endswith('.csv')]
        
        for i in self.topic_list_test:
            self.user_drop_down_select_folder_list_test.insert(tk.END, i)
            
        # Initialise the features_list
        # Extract feature files from the file lists
        # No need to sort the values as the files are already sorted by names
        self.feature_list_test = [x for x in files_tmp_test if x.endswith('_f.pkl')]
        
        for i in self.feature_list_test:
            self.drop_down_list_word_vector_list_test.insert(tk.END, i)
        
        self.concept_list_test = [x for x in files_tmp_test if x.endswith('_c.pkl')]
        
        for i in self.concept_list_test:
            self.drop_down_concept_prob_vector_list_test.insert(tk.END, i)
        
        self.CLDA_list_test = [x for x in files_tmp_test if x.endswith('_CLDA.pkl')]
        for i in self.CLDA_list_test:
            self.drop_down_CLDA_list_test.insert(tk.END, i)
            
        
        self.LDA_list_test = [x for x in files_tmp_test if x.endswith('_LDA.pkl')]
        for i in self.LDA_list_test:
            self.drop_down_LDA_list_test.insert(tk.END, i)
        
            
            

    
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
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
        if(len(for_test_purpose_data) != 0):
            for_test_purpose_data.to_csv(dataset_dir + '/' +
                              topic_name + ".csv",
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + ".csv"


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
        print(training_path)
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
                              topic_name + ".csv",
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
            
            return topic_name + ".csv"
    # Retrieving topic list
    
    # Based on the files_tmp made by the vectors 
    # Depending on what test vector u used 
    # The contents of the vector can be changed 
    
    
    # Asynchoronic feature vector retrieval
    def create_feature_vector_async(self, file_string, feature_list, dataset_dir):
        
        file_string = os.path.splitext(os.path.basename(file_string))[0]
        if any(file_string in substring for substring in feature_list):
            print("Feature {} already exists".format(file_string +  '_f.pkl'))
            return None
        else:
            # Read csv files 
            datum = pd.read_csv(dataset_dir + '/' + file_string + '.csv', encoding='utf-8', sep=',', 
                        error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
            
            # Vectorise the document 
            vect = generate_vector()
            vectorized_data, feature_names = vectorize(vect, datum)
            
            with open(dataset_dir + '/' + file_string + '_f.pkl', "wb") as f:
                pickle.dump([vectorized_data, feature_names], f)
            return file_string + '_f.pkl'
            

    
#   
    def create_concept_matrix_async(self, file_string, concept_list, dataset_dir):

        async def retrieve_word_concept_data(feature_names, K = 20):
            with concurrent.futures.ThreadPoolExecutor(max_workers=150) as executor:
                collection_of_results = []
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
                        requests.get, 
                        'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' +
                        i.replace(' ', '+') + 
                        '&topK=' + str(K)
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
            print("Feature {} already exists".format(file_string +  '_c.pkl'))
        else:
            p_e_c  = {}

            feature_names = None                    
            with open(dataset_dir + '/' + file_string + '_f.pkl', "rb") as f:
                 _, feature_names = pickle.load(f)
            
            
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
            #    print(i)
            #    print(idx)
                p_e_c[i] = results[int(idx)]
            
#                print(p_e_c)
#                temp = {}
#                type(temp) == dict
            # List up the concept names
            
            l = [list(i.keys()) for i in list(p_e_c.values())]
            concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
            
            #    concept_sets[len(concept_sets)-1]
            # Put the atom concept if there are no concepts in the words
            
            
            # Adding atomic elements
            for i in feature_names:
                #if there are no concepts in the words, then...
                if p_e_c[i] == {}:
                    
                    #Append the words with no related concpets
                    #as the atomic concepts
                    concept_names.append(i)
#                    else:
                    
            # Sorting the concept_names after adding feature names
            concept_names = sorted(concept_names)

            
            with open(dataset_dir + '/' + file_string +  '_c.pkl', "wb") as f:
                pickle.dump([p_e_c, concept_names], f)
                
            return file_string +  '_c.pkl'
            
#        for i in self.feature_list:

    # Retrieve all test and training data asynchrously
    
    def asynchronous_topic_concept_retrieval(self, fobj, file_type, dataset_dir, test_or_training):
        train_folder_selection = askdirectory()
        #train_folder_selection = "C:/Users/n9648852/Desktop/R8-Dataset/Dataset/R8/Testing"
        
        training_folders_tmp = []
        
        for dirpath, dirs, files in os.walk(train_folder_selection):
            if len([x for x in files if x.endswith(file_type)]) != 0:
                training_folders_tmp.append(dirpath) 
        
        # training_folders_tmp.remove(train_folder_selection)
        
        if(test_or_training == 0):
            topic_list = self.topic_list
        else:
            topic_list = self.topic_list_test
        '''
        Asynchronous training dataset retrieval
        '''
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
        
        print(topics)
        if None in topics:
            topics = list(filter((None).__ne__, topics))
        print(topics)
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
        
        async def create_feature_vectors(training_folders_tmp, feature_list, dataset_dir):
            
           with concurrent.futures.ThreadPoolExecutor() as executor:
            
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
                        self.create_feature_vector_async, 
                        folder_name, feature_list, dataset_dir
                    )
                    for folder_name in training_folders_tmp
                ]
                features_vec = []
                
                for i in await asyncio.gather(*futures):
                    features_vec.append(i)
                    
                return features_vec
            
        if(test_or_training == 0):
            feature_list = self.feature_list
        else:
            feature_list = self.feature_list_test
        
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(create_feature_vectors(training_folders_tmp, feature_list, dataset_dir))
        features = loop.run_until_complete(future)
        
        
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
        
        
        # Asyncio is not used for retrieving web data simultaneously
        def create_concept_word_lists(training_folders_tmp, concept_list, dataset_dir):
            concept_vec = []
            for i in training_folders_tmp:
                concept_vec.append(self.create_concept_matrix_async(i, concept_list, dataset_dir))
            time.sleep(2)    
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
    
    
    def create_CLDA_instance(i):
        
        CLDA_file_suffix = "_CLDA.pkl"
        file_index_name = pd.read_csv(dataset_dir + '/' + i + '.csv', encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)["File"]
        files = []
        for i,j, k in os.walk(dataset_dir):
            files.extend(k)
        
        if (file_index_name + CLDA_file_suffix in files):
            print("File {} already exists.".format(file_index_name + CLDA_file_suffix))
            return True
        
        feature_matrix, feature_names = (None, None)
        with open( dataset_dir + '/' + i + "_f.pkl", "rb") as f :   
            feature_matrix, feature_names = pickle.load(f)
        
        concept_dict, concept_names = (None, None)
        with open(dataset_dir + '/' + i + "_c.pkl", "rb") as f:
            concept_dict, concept_names = pickle.load(f)
#        
        CLDA_instance = CLDA.CLDA(feature_names, concept_names, file_index_name, 5, 20)
#        print(concept_names)
        CLDA_instance.run(feature_matrix, concept_dict)
        
        with open(dataset_dir + '/' + i + CLDA_file_suffix, "wb") as f:
            pickle.dump(CLDA_instance, f)
        
        # Sleep just in case...
        time.sleep(0.5)
        # Return True if the process stops normally
        return True
        
        
    def asynchronous_CLDA_model_generation(self, dataset_dir, result_num):
        
        def concurrent():
    #        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
            
            fm = Asynchrous_CLDA()
            
            results = fm.asynchronous_CLDA_creation(dataset_dir)
            
            return results
        
        results = concurrent()
        #Create CLDA object asynchronically.
        
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

    def asynchronous_LDA_model_generation(self, dataset_dir, result_num):
        

        def concurrent():
        #        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
            
            fm = Asynchrous_LDA
            
            results = fm.asynchronous_LDA_creation(fm, dataset_dir)
            
            return results
            
        print("LDA model creation start!")
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
        
        
    def create_CLDA_instance(self,i):
        # turn standard out into string IO...
        # to capture the standard output
        sys.stdout = buffer = StringIO()
        # Read CLDA files 
        files_tmp = []
        for dirpath, dirs, files in os.walk(self.dataset_dir):
            if len([x for x in files if x.endswith('.csv')]) != 0:
                files_tmp.extend(files) 
        
        if i + "_CLDA.pkl" in files_tmp:
            print("File {} is already exists".format(i + "_CLDA.pkl"))
            # Normal finish of the program...
            return None, ""
        
        print("file process {}: starts!".format(i))
        file_index_name = pd.read_csv(self.dataset_dir + '/' + i + '.csv', encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)["File"]
        
        feature_matrix, feature_names = (None, None)
        with open( self.dataset_dir + '/' + i + "_f.pkl", "rb") as f :   
            feature_matrix, feature_names = pickle.load(f)
        
        concept_dict, concept_names = (None, None)
        with open(self.dataset_dir + '/' + i + "_c.pkl", "rb") as f:
            concept_dict, concept_names = pickle.load(f)
        print("file reading {}: complete!".format(i))
        sys.stdout.flush()
        CLDA_instance = CLDA.CLDA(feature_names, concept_names, file_index_name, 5, 20)

        CLDA_instance.run(feature_matrix, concept_dict)
        
        with open(self.dataset_dir + '/' + i + "_CLDA.pkl", "wb") as f:
            pickle.dump(CLDA_instance, f)
        
        # Sleep just in case...
        time.sleep(0.5)
        sys.stdout = sys.__stdout__
        
        # Return True if the process stops normally
        return i + "_CLDA.pkl", buffer
    
    def asynchronous_CLDA_creation(self, dataset_dir):
           
        self.dataset_dir = dataset_dir    
        files = []
        for dirpath, dirs, files in os.walk(self.dataset_dir):
            if len([x for x in files if x.endswith('.csv')]) != 0:
                files.extend(files) 
        
        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
        
        # Asynchronically create the CLDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.create_CLDA_instance, [[i] for i in files_list_for_modelling_CLDA])
            
            return pool_async.get()
                
class Asynchrous_LDA(object):
    
    def __init__(self):
        self.__init__(self)
        self.dataset_dir = None
        
        
    def create_LDA_instance(self,i, dataset_dir):
        sys.stdout = buffer = StringIO()  
        files_tmp = []
        for dirpath, dirs, files in os.walk(dataset_dir):
            if len([x for x in files if x.endswith('.csv')]) != 0:
                files_tmp.extend(files) 
        print(dataset_dir + '/' + i + "_LDA.pkl")
        # If the file has already been created 
        # then skip the process...
        if i + "_LDA.pkl" in files_tmp:
            print("File {} is already exists".format(i + "_LDA.pkl"))
            #Normal finish of the program...
            return None, ""
        # Read CLDA files 
        print("file process {}: starts!".format(i))
        file_index_name = pd.read_csv(dataset_dir + '/' + i + '.csv', encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)["File"]
        
        feature_matrix, feature_names = (None, None)
        with open(dataset_dir + '/' + i + "_f.pkl", "rb") as f :   
            feature_matrix, feature_names = pickle.load(f)
        
        
        sys.stdout.flush()
        LDA_instance = CLDA.LDA(file_index_name, feature_names, 5)
        # print(concept_names)
        
        # Later the iteration can be changed
        # from GUI
        LDA_instance.run(feature_matrix, 20)
        
        with open(dataset_dir + '/' + i + "_LDA.pkl", "wb") as f:
            pickle.dump(LDA_instance, f)
        # print("file reading {}: complete!".format(i))
        # Sleep just in case...
        print("Model generation complete!: {}".format(dataset_dir + '/' + i + "_LDA.pkl"))
        time.sleep(0.5)
        # Return True if the process stops normally
        sys.stdout = sys.__stdout__
        return (i + "_LDA.pkl"), buffer
    
    def asynchronous_LDA_creation(self, dataset_dir):
           
          
#        results.insert(tk.END, "ttt")
        files = []
        
        for dirpath, dirs, files in os.walk(dataset_dir):
            if len([x for x in files if x.endswith('.csv')]) != 0:
                files.extend(files) 
        
        # Very rough file detection....
        files_list_for_modelling_LDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
        
        # Core use
        # Asynchronically create the LDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.create_LDA_instance, [[self, i, dataset_dir] for i in files_list_for_modelling_LDA])
            
            return pool_async.get()        
            

        
def main():
    #Run the main application
    Application()


if __name__ == "__main__":
    main()
