# -*- coding: utf-8 -*-
"""
@author: Shotaro Baba
"""
import datetime
import time
import ast
#from multiprocessing import Pool
import pickle
import csv
import numpy as np
import json
import pandas as pd
from nltk.corpus import wordnet as wn
import sys
sys.path.append("..\/models/")
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from multiprocessing import cpu_count
from multiprocessing import Pool

# Setting default values
dataset_dir = "../../data_training"
dataset_test = "../../data_testing"
score_result_dir = "../../score_result"
score_result_dataframe_suffix = "_CLDA_score.csv"
score_result_txt_suffix = "_CLDA_score.txt"
score_result_log_suffix = "_CLDA_log.txt"
LDA_score_result_dataframe_suffix = "_LDA_score.csv"
LDA_score_result_txt_suffix = "_LDA_score.txt"
LDA_score_result_log_suffix = "_LDA_log.txt"
stop_word_folder = "../stopwords"
concept_prob_suffix_json = "_c_prob.json"
concept_name_suffix_txt = "_c_name.txt"
feature_matrix_suffix_csv = "_f_mat.csv"
feature_name_suffix_txt = "_f_name.txt"
file_name_df_suffix_csv = "_data.csv"
tokenized_dataset_suffix= "_tokenized.csv"
CLDA_suffix_pickle = "_CLDA.pkl"
LDA_suffix_pickle = "_LDA.pkl"
converted_xml_suffix = "_conv.txt"
all_score_suffix = "_all_score.csv"
all_score_precision_etc_suffix_LDA = "_LDA_precision_recall_top.csv"
all_score_precision_etc_suffix_CLDA = "_CLDA_precision_recall_top.csv"
suffix_for_testing_csv = "._test_baseline.csv"
suffix_for_testing_txt = "._test_baseline.csv"

# LDA suffix test pickle
LDA_suffix_test_pickle = "_LDA_test.pkl"
default_ngram = 1



delim = ","
default_score_threshold = 0
asterisk_len = 20
default_ranking_show_value = 10

illegal_file_characters = ['+', ',', ';', '=', '[', ']', '\'', '"', '*', '<', '>', '|', '?', '/']

default_rank_concept = 10
default_rank_word = 10
stop_word_folder = "../stopwords"
stop_word_smart_txt = "smart_stopword.txt"
top_10_rank = 10
top_20_rank = 20
smart_stopwords = []

with open(stop_word_folder + '/' + stop_word_smart_txt , "r", encoding='utf8') as f:
    for line in f:
        smart_stopwords.append(line.strip('\n'))



from tkinter.filedialog import asksaveasfilename
import tkinter as tk
import gc
import os
import main_window
dataset_training = "../../data_training"
dataset_testing = "../../data_testing"

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from nltk.util import ngrams 
# initialize constants
lemmatizer = WordNetLemmatizer()


if not os.path.isdir(score_result_dir):
    os.makedirs(score_result_dir)

def define_sw():
    
    # Use default english stopword     
    return set(stopwords.words('english') + smart_stopwords)


#################################
##### Testing purpose
#################################

#def LDA_test_doc_topic_prob(LDA_test, feature_names, top_rank):
##        test = test_LDA_test.components_
#    topic_words = {}
##        prob_words = {}
#    topic_words_prob = {}
#    vocab = feature_names
#    probability = LDA_test.components_ / LDA_test.components_.sum(axis=1)[:, np.newaxis]
#    for topic, comp in enumerate(LDA_test.components_):
#        # for the n-dimensional array "arr":
#        # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"
#        # which contains the indices that would sort arr in a descending fashion
#        # for the ith element in ranked_array, ranked_array[i] represents the index of the
#        # element in arr that should be at the ith index in ranked_array
#        # ex. arr = [3,7,1,0,3,6]
#        # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]
#        # word_idx contains the indices in "topic" of the top num_top_words most relevant
#        # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now) 
#        
#        word_idx = np.argsort(comp)[::-1][:top_rank]
#        normalization = sum([probability[topic][i] for i in word_idx])
#        
#        topic_words[topic] = ["{}: {}".format(vocab[i], probability[topic][i]) for i in word_idx]
#        topic_words_prob[topic] = [(vocab[i], probability[topic][i]/normalization) for i in word_idx]
##        for topic, words in topic_words.items():
##            print('Topic: {}'.format(topic))
##            print('\n'.join(words))
#    return topic_words_prob.items()

#################################
##### Testing purpose
#################################

### The class for displaying popup message
def popupmsg_enter_file_name():
    root_msg = tk.Tk()
    root_msg.title("Warning")
    root_msg.geometry("300x100")
    label = tk.Label(root_msg, text = "Please enter the string")
    label.pack(padx = 10, pady = 10)
    
    button = tk.Button(root_msg, text = "OK", command = root_msg.destroy)
    
    button.pack(padx = 10, pady = 10)
    
    root_msg.mainloop()
#    del root_msg


# The class of CLDA evaluation screen 
class CLDA_evaluation_screen(object):
    
    # Initialise the screen 
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CLDA Evaluation")
        
        
        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="Quit", command=self.root.destroy)
        
        self.root.config(menu=self.menubar)
        
        self.start_menu()
        
        self.listing_all_model_and_result()
        self.input_the_value_first()
        
        self.root.mainloop()
    
    # Showing the initial input values first
    def input_the_value_first(self):
        self.ranking_number_word_textbox.insert(tk.END, "{}".format(default_rank_word))
        self.ranking_number_concept_textbox.insert(tk.END, "{}".format(default_rank_concept))
        self.max_ngram_entry.insert(tk.END, "{}".format(default_ngram))
        self.min_ngram_entry.insert(tk.END, "{}".format(default_ngram))
        self.threshold_entry.insert(tk.END, "{}".format(default_score_threshold))
        self.CLDA_selection_word_ranking_box.insert(tk.END, "{}".format(default_ranking_show_value))
        self.LDA_selection_word_ranking_box.insert(tk.END, "{}".format(default_ranking_show_value))
#        self.
    
    # Going to the start menu
    def start_menu(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()
        
        self.bottom_button_frame = tk.Frame(self.root)
        self.bottom_button_frame.pack()
        
        self.result_and_selection_frame = tk.Frame(self.main_frame)
        self.result_and_selection_frame.pack()
        
        self.selection_of_frame = tk.Frame(self.main_frame)
        self.selection_of_frame.pack(side = tk.LEFT)
        
        self.result_frame = tk.Frame(self.main_frame)
        self.result_frame.pack(side = tk.RIGHT)
        
        '''
        ##########################################################
        ####Text-based result
        ##########################################################
        '''
        #Showing the result of the calculation in the text field
        self.result_frame_main = tk.Frame(self.result_frame)
        self.result_frame_main.pack(pady = 10, padx = 5)
        
        self.result_screen_label = tk.Label(self.result_frame_main, text = "CLDA & LDA Evaluation Result")
        self.result_screen_label.pack()
        
        self.result_screen_text = tk.Text(self.result_frame_main)
        self.result_screen_text.pack(side = tk.LEFT)
        self.result_screen_text.configure(state='disabled')
        
        self.result_screen_text_scroll = tk.Scrollbar(self.result_frame_main)
        self.result_screen_text_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        self.result_screen_text['yscrollcommand'] = \
        self.result_screen_text_scroll.set
        self.result_screen_text_scroll['command'] = self.result_screen_text.yview
        
        
        self.clear_button_frame = tk.Frame(self.result_frame)
        self.clear_button_frame.pack()
        
        self.clear_button = tk.Button(self.clear_button_frame, text = "Clear content")
        self.clear_button.pack()
        self.clear_button['command'] = self.clear_text_result
        
        
        #Selection menu for generating algorithms
        
        '''
        ##########################################################
        ####CLDA model selection
        ##########################################################
        '''
        self.model_selection_frame = tk.LabelFrame(self.selection_of_frame, relief = tk.RAISED, borderwidth = 1, text = "Model Selection")
        self.model_selection_frame.pack(ipadx = 10, ipady = 10)
        
        self.CLDA_selection_and_preference = tk.Frame(self.model_selection_frame, relief = tk.RAISED, borderwidth = 1)
        self.CLDA_selection_and_preference.pack(side = tk.RIGHT, padx = 10)
        
        
        
        self.CLDA_selection_label = tk.Label(self.CLDA_selection_and_preference, text = "CLDA Selection")
        self.CLDA_selection_label.grid(row = 0)
        
        #######################################
        ## Scrolling section
        #######################################
        self.CLDA_selection_and_preference_list_scroll = tk.Frame(self.CLDA_selection_and_preference)
        self.CLDA_selection_and_preference_list_scroll.grid(row = 1)
        
        self.CLDA_selection_listbox = tk.Listbox(self.CLDA_selection_and_preference_list_scroll)
        self.CLDA_selection_listbox.pack(side = tk.LEFT)
        
        self.CLDA_selection_listbox_scroll = tk.Scrollbar(self.CLDA_selection_and_preference_list_scroll)
        self.CLDA_selection_listbox_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        self.CLDA_selection_listbox['yscrollcommand'] = \
        self.CLDA_selection_listbox_scroll.set
        self.CLDA_selection_listbox_scroll['command'] = self.CLDA_selection_listbox.yview
        
        #########################################
        ## Button section
        #########################################
        
        self.CLDA_word_ranking_input_frame = tk.Frame(self.CLDA_selection_and_preference)
        self.CLDA_word_ranking_input_frame.grid(row = 2)
        
        self.CLDA_selection_word_ranking_label = tk.Label(self.CLDA_word_ranking_input_frame, text = "Enter Word\nfor Ranking\nfor CLDA:")
        self.CLDA_selection_word_ranking_label.pack(side = tk.LEFT)
        
        self.CLDA_selection_word_ranking_box = tk.Entry(self.CLDA_word_ranking_input_frame)
        self.CLDA_selection_word_ranking_box.pack(side = tk.RIGHT)
        
        
        self.CLDA_selection_concept_ranking_button = tk.Button(self.CLDA_selection_and_preference, text = "Concept ranking\nevaluation")
        self.CLDA_selection_concept_ranking_button.grid(row = 3)
        self.CLDA_selection_concept_ranking_button['command'] = self.show_CLDA_ranking
        
        self.CLDA_selection_word_ranking_button = tk.Button(self.CLDA_selection_and_preference, text = "Word ranking\nevaluation")
        self.CLDA_selection_word_ranking_button.grid(row = 4)
        self.CLDA_selection_word_ranking_button['command'] = self.show_word_under_concept
        
        self.CLDA_topic_probability_button= tk.Button(self.CLDA_selection_and_preference, text = "Topic probability")
        self.CLDA_topic_probability_button.grid(row = 5)
        self.CLDA_topic_probability_button['command']  = self.show_topic_ranking_CLDA
        
        
        
        ##########################################
        ## Bottom rank section 
        ##########################################
        
        '''
        ##########################################################
        ####LDA model selection
        ##########################################################
        '''
        self.LDA_selection_and_preference = tk.Frame(self.model_selection_frame, relief = tk.RAISED, borderwidth = 1)
        self.LDA_selection_and_preference.pack(side = tk.RIGHT, padx = 10)
        
        
        
        self.LDA_selection_label = tk.Label(self.LDA_selection_and_preference, text = "LDA Selection")
        self.LDA_selection_label.grid(row = 0)
        
        #######################################
        #Scrolling section
        #######################################
        self.LDA_selection_and_preference_list_scroll = tk.Frame(self.LDA_selection_and_preference)
        self.LDA_selection_and_preference_list_scroll.grid(row = 1)
        
        self.LDA_selection_listbox = tk.Listbox(self.LDA_selection_and_preference_list_scroll)
        self.LDA_selection_listbox.pack(side = tk.LEFT)
        
        self.LDA_selection_listbox_scroll = tk.Scrollbar(self.LDA_selection_and_preference_list_scroll)
        self.LDA_selection_listbox_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        #########################################
        ##Button section
        #########################################
        
        self.LDA_word_ranking_input_frame = tk.Frame(self.LDA_selection_and_preference)
        self.LDA_word_ranking_input_frame.grid(row = 2)
        
        self.LDA_selection_word_ranking_label = tk.Label(self.LDA_word_ranking_input_frame, text = "Enter word\nfor ranking\nfor LDA:")
        self.LDA_selection_word_ranking_label.pack(side = tk.LEFT)
        
        self.LDA_selection_word_ranking_box = tk.Entry(self.LDA_word_ranking_input_frame)
        self.LDA_selection_word_ranking_box.pack(side = tk.RIGHT)
        
        
        self.LDA_selection_word_ranking_button = tk.Button(self.LDA_selection_and_preference, text = "Word ranking\nevaluation")
        self.LDA_selection_word_ranking_button.grid(row = 3)
        self.LDA_selection_word_ranking_button['command']  = self.show_LDA_ranking
        
        
        
        self.LDA_topic_probability_button= tk.Button(self.LDA_selection_and_preference, text = "Topic probability")
        self.LDA_topic_probability_button.grid(row = 5)
        self.LDA_topic_probability_button['command']  = self.show_topic_ranking_LDA
        
    
        
        self.LDA_selection_listbox['yscrollcommand'] = \
        self.LDA_selection_listbox_scroll.set
        self.LDA_selection_listbox_scroll['command'] = self.LDA_selection_listbox.yview
        
        
        '''
        ##########################################################
        #### LDA & CLDA word ranking
        ##########################################################
        '''
        self.button_word_ranking_result_frame = tk.Frame(self.selection_of_frame, relief = tk.RAISED, borderwidth = 1)
        self.button_word_ranking_result_frame.pack(side = tk.BOTTOM, padx = 10, ipadx = 10, ipady = 10)
        
        self.button_word_ranking_frame = tk.LabelFrame(self.button_word_ranking_result_frame, relief = tk.RAISED, borderwidth = 1, text = "Model Evaluation Buttons")
        self.button_word_ranking_frame.pack(side = tk.LEFT, padx = 10, ipadx = 10, ipady = 10)

        self.LDA_evaluation_button = tk.Button(self.button_word_ranking_frame, text = "Eval LDA")
        self.LDA_evaluation_button.pack()
        self.LDA_evaluation_button['command'] = self.asynchronous_LDA_evaluation
        
        self.CLDA_evaluation_button = tk.Button(self.button_word_ranking_frame, text = "Eval CLDA")
        self.CLDA_evaluation_button.pack()
        self.CLDA_evaluation_button['command'] = self.asynchronous_CLDA_evaluation
        
#        self.result_save_selection = tk.Button(self.button_word_ranking_frame, text = "Save result")
#        self.result_save_selection.pack()
#        self.CLDA_evaluation_button['command'] = None
        
        self.ranking_label_box_frame = tk.Frame(self.button_word_ranking_frame)
        self.ranking_label_box_frame.pack()
        
        self.ranking_number_concept_label = tk.Label(self.ranking_label_box_frame, text = "Number to retrieve concept: ")
        self.ranking_number_concept_label.grid(row = 0, column=0)
        
        self.ranking_number_concept_textbox = tk.Entry(self.ranking_label_box_frame)
        self.ranking_number_concept_textbox.grid(row = 0, column=1)
        
        self.ranking_number_word_label = tk.Label(self.ranking_label_box_frame, text = "Number to retrieve word: ")
        self.ranking_number_word_label.grid(row = 1, column=0)
        
        self.ranking_number_word_textbox = tk.Entry(self.ranking_label_box_frame)
        self.ranking_number_word_textbox.grid(row = 1, column=1)
        
        self.min_ngram_label = tk.Label(self.ranking_label_box_frame, text = "min_ngram: ")
        self.min_ngram_label.grid(row = 3, column = 0)
        
        self.min_ngram_entry = tk.Entry(self.ranking_label_box_frame)
        self.min_ngram_entry.grid(row = 3, column = 1)
        
        self.max_ngram_label = tk.Label(self.ranking_label_box_frame, text = "max_ngram: ")
        self.max_ngram_label.grid(row = 4, column = 0)
        
        self.max_ngram_entry = tk.Entry(self.ranking_label_box_frame)
        self.max_ngram_entry.grid(row = 4, column = 1)
        
        self.threshold_label = tk.Label(self.ranking_label_box_frame, text = "Threshold (positive value): ")
        self.threshold_label.grid(row = 5, column = 0)
        
        self.threshold_entry = tk.Entry(self.ranking_label_box_frame)
        self.threshold_entry.grid(row = 5, column = 1)
        
#        self.ranking_result_file_name_label = tk.Label(self.ranking_label_box_frame, text = "File name to store the result: ")
#        self.ranking_result_file_name_label.grid(row = 6, column=0)
#        
#        self.ranking_result_file_name_textbox = tk.Entry(self.ranking_label_box_frame)
#        self.ranking_result_file_name_textbox.grid(row = 6, column=1)
        
        '''
        ##########################################################
        #### Result selection section
        ##########################################################
        '''
        self.evaluation_result_list_frame = tk.LabelFrame(self.button_word_ranking_result_frame, relief = tk.RAISED, borderwidth = 1, text = "Evaluation Result List")
        self.evaluation_result_list_frame.pack(side = tk.RIGHT, padx = 10, ipadx = 10, ipady = 10)
        
        self.result_selection_list = tk.Frame(self.evaluation_result_list_frame)
        self.result_selection_list.pack()
        
        self.result_selection_listbox = tk.Listbox(self.result_selection_list)
        self.result_selection_listbox.grid(row = 0, column = 0)
        
        self.result_selection_listbox_scroll_y = tk.Scrollbar(self.result_selection_list)
        self.result_selection_listbox_scroll_y.grid(row = 0, column = 1, sticky  = 'ns')
        
        self.result_selection_listbox_scroll_x = tk.Scrollbar(self.result_selection_list,orient=tk.HORIZONTAL)
        self.result_selection_listbox_scroll_x.grid(row = 1, column = 0, columnspan = 3, sticky  = "we")
        
        self.result_selection_listbox['yscrollcommand'] = \
        self.result_selection_listbox_scroll_y.set
        self.result_selection_listbox_scroll_y['command'] = self.result_selection_listbox.yview
        
        self.result_selection_listbox['xscrollcommand'] = \
        self.result_selection_listbox_scroll_x.set
        self.result_selection_listbox_scroll_x['command'] = self.result_selection_listbox.xview
        
        
        self.displaying_score_button_all  = tk.Button(self.evaluation_result_list_frame, text = "Display results")
        self.displaying_score_button_all.pack()
        self.displaying_score_button_all["command"] = self.display_all_scores_in_file
        
        
        

        '''
        ##########################################################
        #### Return button
        ##########################################################
        '''
        
        
        self.change_to_model_creation = tk.Button(self.bottom_button_frame, text = "Return to model creation")
        self.change_to_model_creation.pack()
        self.change_to_model_creation['command'] = self.move_to_model_creation
    
        
    '''
    ##########################################################
    ##########################################################
    ##########################################################
    ### Retrieve the parameters from input
    ##########################################################
    ##########################################################
    ##########################################################
    '''
    
    # The method for retireving the limit of 
    # top words probablities from either CLDA or LDA over
    # either concept or topic
    def retrieve_top_word_number(self):
        try:
            if(self.ranking_number_word_textbox.get() == ""):
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe default value is used: top rank word num. = {}".format(default_rank_concept))
                self.result_screen_text.configure(state='disabled')
                return default_rank_concept
            else:
                user_input_val = int(self.ranking_number_word_textbox.get())
                if (user_input_val < 1):
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nPlease input the value more than 0! (top rank word num. {})".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe input top rank word num. value {} is used".format(user_input_val))
                self.result_screen_text.configure(state='disabled')
                return user_input_val
        except ValueError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: The input top rank word num. is invalid.")
            self.result_screen_text.configure(state='disabled')
            return
    # The method for retrieving the limit of 
    # top concept probabilities from CLDA over 
    # topic 
    def retrieve_top_concept_number(self):
        try:
            if(self.ranking_number_concept_textbox.get() == ""):
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe default value is used: top rank concept num.  = {}".format(default_rank_concept))
                self.result_screen_text.configure(state='disabled')
                return default_rank_concept
            else:
                user_input_val = int(self.ranking_number_concept_textbox.get())
                if (user_input_val < 1):
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nPlease the value more than 1! (top rank concept num.  {})".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe input min top rank concept num. value {} is used".format(user_input_val))
                self.result_screen_text.configure(state='disabled')
                return user_input_val
        except ValueError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: The input top rank concept num. is invalid.")
            self.result_screen_text.configure(state='disabled')
            return
    # Retrieve the min ngram number 
    # from entry 
    def retrieve_ngram_min(self):
        try:
            if(self.min_ngram_entry.get() == ""):
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe default value is used: ngram = {}".format(default_ngram))
                self.result_screen_text.configure(state='disabled')
                return default_ngram
            else:
                user_input_val = int(self.min_ngram_entry.get())
                if(user_input_val < 1):
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nInput positive value! ngram min {}".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return 
                else:
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nThe input ngram value {} is used".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: The input ngram value is invalid.")
            self.result_screen_text.configure(state='disabled')
            return
    # Retrieve the max ngram number 
    # from entry 
    def retrieve_ngram_max(self):
        try:
            if(self.max_ngram_entry.get() == ""):
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe default value is used: threshold = {}".format(default_ngram))
                self.result_screen_text.configure(state='disabled')
                return default_ngram
            else:
                user_input_val = int(self.max_ngram_entry.get())
                if(user_input_val < 1):
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nInput positive value! mgram max {}".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return 
                else:
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nThe input threshold value {} is used".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: The input threshold value is invalid.")
            self.result_screen_text.configure(state='disabled')
            return
    # Retrieve threshold value  
    # from entry 
    def retrieve_threshold(self):
        try:
            if(self.threshold_entry.get() == ""):
                self.result_screen_text.configure(state='normal')
                self.result_screen_text.insert(tk.END, "\nThe default value is used: beta = {}".format(default_score_threshold))
                self.result_screen_text.configure(state='disabled')
                return default_score_threshold
            else:
                user_input_val = float(self.threshold_entry.get())
                if(user_input_val < 0):
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nInput positive or zero value! threshold: {}".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return 
                else:
                    self.result_screen_text.configure(state='normal')
                    self.result_screen_text.insert(tk.END, "\nThe input beta value {} is used".format(user_input_val))
                    self.result_screen_text.configure(state='disabled')
                    return user_input_val
        
        except ValueError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: The input max iter. is invalid.")
            self.result_screen_text.configure(state='disabled')
    '''
    ##########################################################
    ##########################################################
    ##########################################################
    ### End of retrieve the parameters from input
    ##########################################################
    ##########################################################
    ##########################################################
    '''



    # Save the results in user selected folder and 
    # the directory...
    # If the directory and store is not selected...
    # then the results will be stored in default 
    # results folders
    def _save_results(self, score_data_frame, all_score_data, result_str, result_log, 
                     score_result_dataframe_suffix, 
                        score_result_txt_suffix, 
                        score_result_log_suffix,
                        all_score_precision_etc_suffix):
        # Extract the file name
        raw_file_name = ""
        while True:
            raw_file_name = asksaveasfilename(title='Please set your result file name')
            if(raw_file_name == ""):
                popupmsg_enter_file_name()
            else:
                break
        # Extract the file name
        file_name = os.path.basename(raw_file_name)
        
        # Extract the folder name
        dir_name = os.path.dirname(raw_file_name)
        
        # Choosing the file name for storing result data       
        storing_result_name_data = dir_name + '/' + file_name + score_result_dataframe_suffix
        
        # Choosing the result name for sotring result data
        storing_result_name_text = dir_name + '/' + file_name + score_result_txt_suffix
        
        # Making the log file name
        log_txt_name = dir_name + '/' + file_name + score_result_log_suffix
        
        # Making the score file name
        all_score_file_name = dir_name + '/' + file_name + all_score_precision_etc_suffix
        
        # The score data is saved as csv file
        score_data_frame.to_csv(storing_result_name_data,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
        
        # The score data is store as the csv
        all_score_data.to_csv(all_score_file_name,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
        
        # The score data is stored
        with open(storing_result_name_text, 'w') as f:
            f.write(result_str)
        
        # The log file is stored in the 
        # text files
        with open(log_txt_name, 'w') as f:
            f.write(result_log)
        
        return file_name
    
    # Move to main window 
    # and destroy the current object
    def move_to_model_creation(self):
        
        # Erase all the display in the directory
        self.root.destroy()
        
        # Delete the object to erase its data
        del self
        
        # Conduct garbage collection after deleting the object 
        # to make free space
        gc.collect()
        
        # Going to main window
        main_window.main()
    
    # Show word probability ranking over 
    # topics
    def show_LDA_ranking(self):
        
        # The output 
        def output():
            sys.stdout = buffer = StringIO()
            ranking_value = default_ranking_show_value
            
            # Retrieve the name of LDA from LDA listbox
            try:
                topic_name = self.LDA_selection_listbox.get(self.LDA_selection_listbox.curselection())
            except:
                print("LDA is not selected")
                sys.stdout = sys.__stdout__
                return buffer
            
            rank_temp = self.LDA_selection_word_ranking_box.get()
#            print(self.LDA_selection_word_ranking_box.get())
            
            # Retrieve teh ranking values from 
            # LDA selection word ranking box
            try:
                ranking_value = int(rank_temp)
                if(ranking_value < 1):
                    print("The value is smaller than 1")
                    ranking_value = default_ranking_show_value
                    print("The default value {} is used".format(ranking_value))
            except ValueError:
                print("Cannot convert the value.")
                print("Default value {} is used.".format(ranking_value))
            
            # Reading the LDA pickle file to read
            topic_name = self.LDA_selection_listbox.get(self.LDA_selection_listbox.curselection())
            
            print("*"*10)
            print(topic_name)
            print("*"*10)
            # The suffix is removed 
            topic_name = topic_name[:-len(LDA_suffix_pickle)]
            
            # The data will be read by the LDA
            with open(dataset_dir + '/' + topic_name + LDA_suffix_pickle, "rb") as f:
                test_LDA = pickle.load(f)
                
            test_LDA.show_word_topic_ranking(ranking_value)
            
            
            sys.stdout = sys.__stdout__
            return buffer
    
        output_buffer = output()
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, output_buffer.getvalue())
        self.result_screen_text.configure(state='disabled')
    
    # Show the average topic probability of 
    # all documents
    def show_topic_ranking_LDA(self):
        def output():
            sys.stdout = buffer = StringIO()
            
            # Extract the selected LDA name from the listbox
            try:
                topic_name = self.LDA_selection_listbox.get(self.LDA_selection_listbox.curselection())
            except:
                print("LDA is not selected")
                sys.stdout = sys.__stdout__
                return buffer
            
            print("*"*10)
            print(topic_name)
            print("*"*10)
            # Topic name extraction
            topic_name = topic_name[:-len(LDA_suffix_pickle)]
            
            # Load the data of LDA
            with open(dataset_dir + '/' + topic_name + LDA_suffix_pickle, "rb") as f:
                test_LDA = pickle.load(f)  
            
            # Show the topic model average probablity 
            test_LDA.show_doc_topic_average_prob()
            
            
            sys.stdout = sys.__stdout__
            return buffer
    
        output_buffer = output()
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, output_buffer.getvalue())
        self.result_screen_text.configure(state='disabled')
    
    # Show average topic probabilities
    def show_topic_ranking_CLDA(self):
        
        def output():
            sys.stdout = buffer = StringIO()
            
            # The selected topic model in the listbox is read 
            try:
                topic_name = self.CLDA_selection_listbox.get(self.CLDA_selection_listbox.curselection())
            except:
                print("CLDA is not selected")
                sys.stdout = sys.__stdout__
                return buffer
            
            
            print("*"*10)
            print(topic_name)
            print("*"*10)
            
            topic_name = topic_name[:-len(CLDA_suffix_pickle)]
          
            with open(dataset_dir + '/' + topic_name + CLDA_suffix_pickle, "rb") as f:
                test_CLDA = pickle.load(f)  
                
            test_CLDA.show_doc_topic_average_prob()
            
            
            sys.stdout = sys.__stdout__
            return buffer
    
        output_buffer = output()
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, output_buffer.getvalue())
        self.result_screen_text.configure(state='disabled')
    
    # Show CLDA topic ranking 
    def show_CLDA_ranking(self):
        def output():
            
            sys.stdout = buffer = StringIO()
            ranking_value = default_ranking_show_value
            
            # The selected topic model in the listbox is read 
            try:
                topic_name = self.CLDA_selection_listbox.get(self.CLDA_selection_listbox.curselection())
            except:
                print("CLDA is not selected")
                sys.stdout = sys.__stdout__
                return buffer
            
            
            rank_temp = self.CLDA_selection_word_ranking_box.get()
            
            # The ranking values are retrieved from the 
            # ranking values
            try:
                ranking_value = int(rank_temp)
                if(ranking_value < 1):
                    print("The value is smaller than 1")
                    ranking_value = default_ranking_show_value
                    print("The default value {} is used".format(ranking_value))
            except ValueError:
                print("Cannot convert the value.")
                print("Default value {} is used.".format(ranking_value))
            
            print("*"*10)
            print(topic_name)
            print("*"*10)
            topic_name = topic_name[:-len(CLDA_suffix_pickle)]
                
            with open(dataset_dir + '/' + topic_name + CLDA_suffix_pickle, "rb") as f:
                test_CLDA = pickle.load(f)
                
            test_CLDA.show_concept_topic_ranking(ranking_value)
            
            
            sys.stdout = sys.__stdout__
            
            return buffer
    
        output_buffer = output()
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, output_buffer.getvalue())
        self.result_screen_text.configure(state='disabled')
    
    # Show word probabilities under concept 
    def show_word_under_concept(self):
        # show_word_concept_prob
        def output():
            
            # Monitoring & taking the standard output 
            # from buffer
            sys.stdout = buffer = StringIO()
            
            ranking_value = default_ranking_show_value
            
            # Make the selection of the topic of CLDA
            try:
                topic_name = self.CLDA_selection_listbox.get(self.CLDA_selection_listbox.curselection())
            except:
                print("CLDA is not selected")
                sys.stdout = sys.__stdout__
                return buffer

            rank_temp = self.CLDA_selection_word_ranking_box.get()
            
            # Try to extract the values in the string 
            try:
                ranking_value = int(rank_temp)
                if(ranking_value < 1):
                    print("The value is smaller than 1")
                    ranking_value = default_ranking_show_value
                    print("The default value {} is used".format(ranking_value))
            except ValueError:
                print("Cannot convert the value.")
                print("Default value {} is used.".format(ranking_value))
            
            # Extract the topic name from the file
            topic_name = self.CLDA_selection_listbox.get(self.CLDA_selection_listbox.curselection())
            print("*"*10)
            print(topic_name)
            print("*"*10)
            # Extract only topic name
            topic_name = topic_name[:-len(CLDA_suffix_pickle)]
            
            with open(dataset_dir + '/' + topic_name + concept_prob_suffix_json, "r") as f:
                test_concept_prob = json.load(f)
                    
            with open(dataset_dir + '/' + topic_name + CLDA_suffix_pickle, "rb") as f:
                test_CLDA = pickle.load(f)
                
            test_CLDA.show_word_concept_prob(test_concept_prob, ranking_value)
            
            
            sys.stdout = sys.__stdout__
            return buffer
        
        output_buffer = output()
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, output_buffer.getvalue())
        self.result_screen_text.configure(state='disabled')
        
    def clear_text_result(self):
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.delete("1.0",tk.END)
        self.result_screen_text.configure(state='disabled')
    
    # List all files generated during 
    # the process
    def listing_all_model_and_result(self):
        files_tmp = []
        
        # Initalise all the contents of the listboxes
        self.CLDA_selection_listbox.delete(0,tk.END)
        self.LDA_selection_listbox.delete(0,tk.END)
        self.result_selection_listbox.delete(0,tk.END)
        
        for dirpath, dirs, files in os.walk(dataset_training):
            files_tmp.extend(files)
        
        self.LDA_list = [x for x in files_tmp if x.endswith("_LDA.pkl")]
        
        # Put all generated LDA models
        for i in self.LDA_list:
            self.LDA_selection_listbox.insert(tk.END, i)

        self.CLDA_list = [x for x in files_tmp if x.endswith("_CLDA.pkl")]
        
        # Put all generated CLDA models
        for i in self.CLDA_list:
            self.CLDA_selection_listbox.insert(tk.END, i)
        
        files_tmp = []
        for dirpath, dirs, files in os.walk(score_result_dir):
            files_tmp.extend(files)
         
        # Put all the results files into 
        # Listbox 
        self.result_list = [x for x in files_tmp if x.endswith(all_score_precision_etc_suffix_CLDA) or x.endswith(all_score_precision_etc_suffix_LDA)]
        
        for i in self.result_list:
            self.result_selection_listbox.insert(tk.END, i)
    
    # Asynchronous CLDA evaluation 
    def asynchronous_CLDA_evaluation(self):
        # Initialize file_tmp list
        # Putting ngram value
        ngram_num_min = self.retrieve_ngram_min()
        if (ngram_num_min == None):
            return
        # If the input value is invalid, then the process
        # will stop
        ngram_num_max = self.retrieve_ngram_max()
        if (ngram_num_max == None):
            return
        # If the input value is invalid, then the process
        # will stop
        threshold_value = self.retrieve_threshold()
        if(threshold_value == None):
            return
        
        self.result_screen_text.delete("1.0", tk.END)
        # If the input value is invalid, then the process
        # will stop
        top_word_number = self.retrieve_top_word_number()
        if top_word_number == None:
            return
        print(top_word_number)
        # If the input value is invalid, then the process
        # will stop
        top_concept_number = self.retrieve_top_concept_number()
        if top_concept_number == None:
            return
        print(top_concept_number)
        
        if(ngram_num_max <  ngram_num_min):
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: max_ngram < min_ngram is not accepted")
            self.result_screen_text.configure(state='disabled')
            return
        
  
        
        
        files_training = []
        for dirpath, dirs, files in os.walk(dataset_dir):
                files_training.extend(files)

#        score_list = []
        files_test = []
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_test.extend(files)
        

        #Tokenize all data in test dataset
        def concurrent1():
        #        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
            
            fm = Asynchronous_CLDA_evaluation_class(rank_concept = top_concept_number, rank_word = top_word_number)
            
            results = fm.asynchronous_tokenization(ngram_num_min, ngram_num_max)
            
            return results
        
        testing_dict = concurrent1()

        # Evaluate all CLDA model 
        def concurrent2():
        #        files_list_for_modelling_CLDA = sorted(list(set([os.path.splitext(x)[0] for x in files if x.endswith('.csv')])))
            
            fm = Asynchronous_CLDA_evaluation_class(rank_concept = top_concept_number, rank_word = top_word_number)
            
            results = fm.asynchronous_evaluation(testing_dict)
            
            return results
        # Generate score
        score_buffers_alpha_beta = concurrent2()
        # Return processed result
        score_result_total = []
        result_log = ""
        
        # The score results are stored in the score log files
        for score_i, buffer_str, alpha, beta in score_buffers_alpha_beta:
            score_result_total.extend(score_i)
            result_log += "\nalpha: {}".format(alpha) + "\nbeta: {}".format(beta) + buffer_str.getvalue()
#        print(score_result_total)  
        
        self._generate_score(score_result_total, result_log, threshold = threshold_value)

    
    def asynchronous_LDA_evaluation(self):
        # If the input value is invalid, then the process
        # stops
        top_word_number = self.retrieve_top_word_number()
        if top_word_number == None:
            return

        # If the input value is invalid, then the process
        # stops
        ngram_num_min = self.retrieve_ngram_min()
        if (ngram_num_min == None):
            return
        # If the input value is invalid, then the process
        # stops
        ngram_num_max = self.retrieve_ngram_max()
        if (ngram_num_max == None):
            return
        # If the input value is invalid, then the process
        # stops
        threshold_value = self.retrieve_threshold()
        if(threshold_value == None):
            return
        # If the input value is invalid, then the process
        # stops
        self.result_screen_text.delete("1.0", tk.END)
        files_training = []
        for dirpath, dirs, files in os.walk(dataset_dir):
                files_training.extend(files)
        
        if(ngram_num_max <  ngram_num_min):
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "\nError: max_ngram < min_ngram is not accepted")
            self.result_screen_text.configure(state='disabled')
            return
        
        files_test = []
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_test.extend(files)
        

        # Tokenize all data in test dataset
        def concurrent1():

            
            fm = Asynchronous_CLDA_evaluation_class(rank_word = top_word_number)
            
            results = fm.asynchronous_tokenization(ngram_num_min, ngram_num_max)
            
            return results
        
        testing_dict = concurrent1()

        # Evaluate all LDA model performances 
        def concurrent2():

            
            fm = Asynchronous_CLDA_evaluation_class(rank_word = top_word_number)
            
            results = fm.asynchronous_evaluation_LDA(testing_dict)
            
            return results
        
        score_buffers_alpha_beta = concurrent2()
        # Return processed result
        score_result_total = []
        result_log = ""
        
        for score_i, buffer_str, alpha, beta in score_buffers_alpha_beta: #score_i_test
            score_result_total.extend(score_i)
            
            
            result_log += "\nalpha: {}".format(alpha) + "\nbeta: {}".format(beta) + buffer_str.getvalue()

        
        # Generate score
        self._generate_score(score_result_total,result_log, 
                             threshold_value,
                             LDA_score_result_dataframe_suffix,
                             LDA_score_result_txt_suffix, 
                             LDA_score_result_log_suffix, all_score_precision_etc_suffix_LDA)
        
#        
    # Calculate the score from the
    # obtained result
    def _generate_score(self, score_result_total,
                        result_log,
                        threshold = default_score_threshold, 
                        score_result_dataframe_suffix = score_result_dataframe_suffix, 
                        score_result_txt_suffix = score_result_txt_suffix, 
                        score_result_log_suffix = score_result_log_suffix,
                        all_score_precision_etc_suffix = all_score_precision_etc_suffix_CLDA):
        ts = time.time()
        

        # Turn the score_result_total into data frame
        score_data_frame = pd.DataFrame(score_result_total, columns = ['File_name', 'Training_topic', 'Testing_topic', 'Score', 'Label', 'alpha', 'beta'])
        

        
        # Obtain file name from entrybox
#        file_name_for_result = self.ranking_result_file_name_textbox.get()
        all_score_data_list = []
        
        
        
        
        # Counting TP, FN, TN and FP 
        Total_TP_count = pd.Series(np.where((score_data_frame['Score'] > threshold) & (score_data_frame['Label'] == 1), True, False)).sum()
        Total_FN_count = pd.Series(np.where((score_data_frame['Score'] <= threshold) & (score_data_frame['Label'] == 1), True, False)).sum()
        Total_TN_count = pd.Series(np.where((score_data_frame['Score'] <= threshold) & (score_data_frame['Label'] == 0), True, False)).sum()
        Total_FP_count = pd.Series(np.where((score_data_frame['Score'] > threshold) & (score_data_frame['Label'] == 0), True, False)).sum()
        
        # Calculate precision and recall and F1.
        precision = Total_TP_count / (Total_TP_count + Total_FP_count)
        recall = Total_TP_count / (Total_TP_count + Total_FN_count)
        
        Total_F1_value = 2 * (precision * recall) / (precision + recall)
        
        local_result_str = ""
        
        #Calculate each accuracy of either LDA or CLDA accuracies
        for i in score_data_frame.Training_topic.unique():
            
            score_data_frame_local = score_data_frame[score_data_frame['Training_topic'] == i]
            score_data_frame_local =  score_data_frame_local.sort_values(by = ['Score'], ascending = False)
            alpha = score_data_frame[score_data_frame['Training_topic'] == i].iloc[0, 5]
            beta = score_data_frame[score_data_frame['Training_topic'] == i].iloc[0, 6]
            
            local_TP_count = pd.Series(np.where((score_data_frame_local['Score'] > threshold) & (score_data_frame_local['Label'] == 1), True, False)).sum()
            local_FN_count = pd.Series(np.where((score_data_frame_local['Score'] <= threshold) & (score_data_frame_local['Label'] == 1), True, False)).sum()
            
            
            local_TN_count = pd.Series(np.where((score_data_frame_local['Score'] <= threshold) & (score_data_frame_local['Label'] == 0), True, False)).sum()
            local_FP_count = pd.Series(np.where((score_data_frame_local['Score'] > threshold) & (score_data_frame_local['Label'] == 0), True, False)).sum()
                
            local_precision = local_TP_count / (local_TP_count + local_FP_count)
            local_recall = local_TP_count / (local_TP_count + local_FN_count)
            local_F1_value = 2 * (local_precision * local_recall) / (local_precision + local_recall)
            
            top_10_rank_precision = score_data_frame_local.head(top_10_rank)['Label'].sum() / float(top_10_rank)
            top_20_rank_precision = score_data_frame_local.head(top_20_rank)['Label'].sum() / float(top_20_rank)
            
            local_result_str += "Topic {}: Precision: {}, Recall: {}\n".format(i, local_precision, local_recall) + \
            "F1 value: {}\n".format(local_F1_value) + \
            "Trup Positive: {}, False Positive: {}\n".format(local_TP_count, local_FP_count) + \
            "Trup Negative: {}, False Negative: {}\n".format(local_TN_count, local_FN_count) + \
            "Top 10 retrieval rate: {}, Top 20 retrieval rate: {}\n".format(top_10_rank_precision, top_20_rank_precision) + \
            "alpha: {}, beta: {}".format(alpha, beta)
            all_score_data_list.append((i, local_precision, local_recall, local_F1_value,  top_10_rank_precision, top_20_rank_precision, alpha, beta, threshold))
        
        all_score_data_list.append(("Total", precision,
                                    recall,
                                    Total_F1_value,
                                    np.nansum([x[4] for x in all_score_data_list])/len(all_score_data_list),
                                    np.nansum([x[5] for x in all_score_data_list])/len(all_score_data_list), "None", "None", threshold))
        
        
        #Storing all necessary data
        storing_result_name_data = ""
        storing_result_name_text = ""
        log_txt_name = ""
        
        # Create all_score_data dataframe from all_score_data_list 
        all_score_data = pd.DataFrame(all_score_data_list, columns= ["Topic", "Precision", "Recall", "F1", "Top10", "Top20", "alpha", "beta", "threshold"])
        
        # Construct result string
        result_str = "All average: Precision: {}, Recall: {}\n".format(precision, recall) + \
            "F1 value: {}\n".format(Total_F1_value) + \
            "Trup Positive: {}, False Positive: {}\n".format(Total_TP_count, Total_FP_count) + \
            "Trup Negative: {}, False Negative: {}\n\n".format(Total_TN_count, Total_FN_count) + local_result_str
        
        file_name_for_result = self._save_results(score_data_frame, all_score_data, result_str, result_log,
                   score_result_dataframe_suffix,
                   score_result_txt_suffix,
                   score_result_log_suffix,
                   all_score_precision_etc_suffix)
        
        # If the file name is invalid, then 
        # the file name is replaced by timestamp
        if(any([x in file_name_for_result for x in illegal_file_characters]) ):
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "".join(['*' for m in range(asterisk_len)]) + '\n')
            self.result_screen_text.insert(tk.END, "Filename is invalid or not input, the time_stamp is used for storing file.")
            self.result_screen_text.insert(tk.END, "".join(['*' for m in range(asterisk_len)]))
            self.result_screen_text.configure(state='disabled')
            
            storing_result_name_data = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_dataframe_suffix
            storing_result_name_text = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_txt_suffix
            log_txt_name = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_log_suffix
            all_score_file_name = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + all_score_precision_etc_suffix
            
        # If the file name is empty, then 
        # the file name is replaced by timestamp
        elif(file_name_for_result == ""):
            storing_result_name_data = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_dataframe_suffix
            storing_result_name_text = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_txt_suffix
            log_txt_name = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + score_result_log_suffix
            all_score_file_name = score_result_dir + '/' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S') + all_score_precision_etc_suffix
        
        # If the file name is valid, then 
        # the file name is used
        else:
            storing_result_name_data = score_result_dir + '/' + file_name_for_result + score_result_dataframe_suffix
        
            storing_result_name_text = score_result_dir + '/' + file_name_for_result + score_result_txt_suffix
            log_txt_name = score_result_dir + '/' + file_name_for_result + score_result_log_suffix
            all_score_file_name = score_result_dir + '/' + file_name_for_result + all_score_precision_etc_suffix
            
        
        # The score data is saved as csv file
        score_data_frame.to_csv(storing_result_name_data,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
        
        
        
        # Input the all scores of models 
        sys.stdout = dataframe_buffer = StringIO()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(all_score_data.to_string(index=False))
        sys.stdout = sys.__stdout__
        
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, "".join(['*' for m in range(asterisk_len)]) + '\n')
        self.result_screen_text.insert(tk.END, dataframe_buffer.getvalue())
        self.result_screen_text.insert(tk.END, "".join(['*' for m in range(asterisk_len)]))
        self.result_screen_text.configure(state='disabled')
        
        
        all_score_data.to_csv(all_score_file_name,
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
        
            
        
        # Storing the results containing the 
        # results values 
        with open(storing_result_name_text, 'w') as f:
            f.write(result_str)
        
        # Write down the log of results in the 
        # string 
        with open(log_txt_name, 'w') as f:
            f.write(result_log)
        
        # Save files into a certain folders
        # and 

        
        self.listing_all_model_and_result()
    
    # The function for inputting results into the results outputs  
    def insert_result_string(self, text):
        self.result_screen_text.configure(state='normal')
        self.result_screen_text.insert(tk.END, text)
        self.result_screen_text.configure(state='disabled')
    
    def display_all_scores_in_file(self):

        
        try:
            file_name = self.result_selection_listbox.get(self.result_selection_listbox.curselection())
        except:
            self.insert_result_string("\nThe result is not selected.")
            return
        
        file_name = score_result_dir + '/' + file_name
        
        score_data_frame = pd.read_csv(file_name,
                                       encoding='utf-8',
                                       quoting=csv.QUOTE_ALL)
        try:

            sys.stdout = score_data_buffer = StringIO()
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(score_data_frame.to_string(index = False))
            sys.stdout = sys.__stdout__
            self.insert_result_string(score_data_buffer.getvalue())
        except IndexError  or pd.errors.ParserError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "Illegal file format.")
            self.result_screen_text.configure(state='disabled')
        except KeyError:
            self.result_screen_text.configure(state='normal')
            self.result_screen_text.insert(tk.END, "The selected model with the topic does not exist.")
            self.result_screen_text.configure(state='disabled')
            
# Asynchronous CLDA evaluation class
# This class also includes LDA evaluation 
# classes
class Asynchronous_CLDA_evaluation_class():

    
    def __init__(self, rank_concept = default_rank_concept, rank_word = default_rank_word):
        self.rank_concept = rank_concept
        self.rank_word = rank_word
        pass
    
    
    # The tokenization method must be the same!
    # initialize constants
    lemmatizer = WordNetLemmatizer()
    def define_sw(self):
        
        # Use default english stopword     
        return set(stopwords.words('english') + smart_stopwords)


    # Defining the lemmatizer
    def lemmatize(self,token, tag):
        tag = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }.get(tag[0], wordnet.NOUN)
    
        return lemmatizer.lemmatize(token, tag)
    
    # The tokenizer for the documents
    def cab_tokenizer(self, document, min_ngram, max_ngram):
        tokens = []
        sw = self.define_sw()
        punct = set(punctuation)
        final_tokens = []
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
                lemma = self.lemmatize(token, tag)
    
                # Append lemmatized token to list
                tokens.append(lemma)
        
        if(min_ngram == 1 and max_ngram == 1):
            return list(set(tokens))
        
        for i in range(min_ngram, max_ngram + 1):
            final_tokens.extend([' '.join(x) for x in list(ngrams(tokens, i))])
            
        final_tokens = list(set(final_tokens))
        # Eliminate duplicates
        return final_tokens
    
    
    # Evaluate all CLDA models
    def calculate_score_all_async(self, training_file_head, testing_dict):
        # Listing the score
        sys.stdout = buffer = StringIO() 
        score_list = []
        files_test = []
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_test.extend(files)
            
        # Extracting numbers 
        training_head_number = ''.join(filter(str.isdigit, training_file_head))
        
        
        testing_file_head = [x[:-len(file_name_df_suffix_csv)] for x in files_test if x.endswith(file_name_df_suffix_csv) and 
                       training_head_number == ''.join(filter(str.isdigit, x))][0]
          
        print("".join(['*' for m in range(asterisk_len)]))
        print("Training_topic: {}".format(training_file_head))
        print("".join(['*' for m in range(asterisk_len)]))
        
        test_concept_prob = None
        
        # Read concept files         
        with open(dataset_dir + '/' + training_file_head + concept_prob_suffix_json, "r") as f:
            test_concept_prob = json.load(f)
            
        
        # Load CLDA model
        with open(dataset_dir + '/' + training_file_head + CLDA_suffix_pickle, "rb") as f:
            test_CLDA = pickle.load(f)
          
        # Calculate the average of topic probabilities 
        doc_topic = test_CLDA.nmz.sum(axis = 0) + test_CLDA.alpha
        doc_topic /= np.sum(doc_topic)
        topic_concept = test_CLDA.construct_normalized_concept_topic_ranking(self.rank_concept)
        word_under_concept = test_CLDA.construct_word_concept_prob_under_concept(test_concept_prob, self.rank_word)

        alpha = test_CLDA.alpha
        beta = test_CLDA.beta
    
        print("".join(['*' for m in range(asterisk_len)]))
        print("Test topic: {}".format(testing_file_head))
        print("".join(['*' for m in range(asterisk_len)]))
        
        # The tokens of test file data is 
        # assigned
        test_file_data = testing_dict[testing_file_head]
        testing_head_number = ''.join(filter(str.isdigit, testing_file_head))
        for i in range(len(test_file_data)):
            score = 0
            test_files_feature_name  = test_file_data.iloc[i]['Text']
            found_topic_concept_word = []
            
            # Calculate the sum of all p(c|z)p(w|c)p(z) if the word w under concept
            # c under topic z is in doucment d
            for topic_num, topic_prob in enumerate(doc_topic):
                for concept, concept_prob in topic_concept[topic_num]:
                    for word, word_prob in [(x[0], x[2]) for x in word_under_concept if x[1] == concept]:
                        if word in test_files_feature_name:
                            score += topic_prob * concept_prob * word_prob
                            found_topic_concept_word.append((topic_num, concept, concept_prob,  word, word_prob))
            
            # All results are printed and 
            # its standard output will be stored in buffer            
            print('File name: "{}", Training_Topic {}, Test_topic {}, Score: {}, Label: {}, topic, word and concept: {}'.format(test_file_data.iloc[i]['File'],
                  training_head_number,
                  testing_head_number,
                  score, test_file_data.iloc[i]['label'], found_topic_concept_word))
            
            # The score is stored as a list.
            score_list.append((test_file_data.iloc[i]['File'], training_head_number, testing_head_number, score, test_file_data.iloc[i]['label'], alpha, beta))
        print("".join(['*' for m in range(asterisk_len)]))
        print("".join(['*' for m in range(asterisk_len)]))
        sys.stdout = sys.__stdout__
        return (score_list, buffer, alpha, beta)
    
    def asynchronous_evaluation(self, testing_dict):
        files_training = []
        for dirpath, dirs, files in os.walk(dataset_dir):
                files_training.extend(files)
        
        # Walk down the files to search for
        # files to geenrate model
        training_head = [x[:-len(file_name_df_suffix_csv)] for x in files_training if x.endswith(file_name_df_suffix_csv)]
        
        
        
        # Core use
        # Asynchronically create the LDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.calculate_score_all_async, [[i, testing_dict] for i in training_head])
            gc.collect()
            # Return processed result
            return pool_async.get()
    
    def asynchronous_evaluation_LDA(self, testing_dict):
        files_training = []
        for dirpath, dirs, files in os.walk(dataset_dir):
                files_training.extend(files)
        
        # Walk down the files to search for
        # files to geenrate model
        training_head = [x[:-len(file_name_df_suffix_csv)] for x in files_training if x.endswith(file_name_df_suffix_csv)]
        # Core use
        # Asynchronically create the LDA object
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.calculate_score_all_async_LDA, [[i, testing_dict] for i in training_head])
            gc.collect()
            # Return processed result
            return pool_async.get()
        
    # Calculate the score of LDA
    def calculate_score_all_async_LDA(self, training_file_head, testing_dict):
        sys.stdout = buffer = StringIO() 
        score_list = []
        files_test = []
        for dirpath, dirs, files in os.walk(dataset_test):
            files_test.extend(files)
        
        # Extract the number of LDA
        training_head_number = ''.join(filter(str.isdigit, training_file_head))
#        sys.stdout = sys.__stdout__
#        print(training_head_number)
        testing_file_head = [x[:-len(file_name_df_suffix_csv)] for x in files_test if x.endswith(file_name_df_suffix_csv) and 
                       training_head_number == ''.join(filter(str.isdigit, x))][0]
        
#        testing_file_head = training_head_number
        print("".join(['*' for m in range(asterisk_len)]))
        print("Training_topic: {}".format(training_file_head))
        print("".join(['*' for m in range(asterisk_len)]))
            
        # Load LDA model
        with open(dataset_dir + '/' + training_file_head + LDA_suffix_pickle, "rb") as f:
            test_LDA = pickle.load(f)
          
        # Calculate topic probabilities
        doc_topic = test_LDA.nmz.sum(axis = 0) + test_LDA.alpha
        doc_topic /= np.sum(doc_topic)
        # Extract rank  = 10
        word_topic_prob = test_LDA.generate_word_prob(self.rank_word)
        
        # Assign alpha and beta value of LDA and CLDA 
        alpha = test_LDA.alpha
        beta = test_LDA.beta
        
        test_file_data = testing_dict[testing_file_head]
        testing_head_number = ''.join(filter(str.isdigit, testing_file_head))
        print("".join(['*' for m in range(asterisk_len)]))
        print("Test topic: {}".format(testing_file_head))
        print("".join(['*' for m in range(asterisk_len)]))

        for i in range(len(test_file_data)):
            score = 0
            test_files_feature_name  = test_file_data.iloc[i]['Text']
            found_word = []
            
            for topic_num, topic_prob in enumerate(doc_topic):
                for word, word_prob in [(x[1], x[2]) for x in word_topic_prob if x[0] == topic_num]:
                    if word in test_files_feature_name:
                        score += topic_prob * word_prob
                        found_word.append((topic_num, topic_prob, word, word_prob))
            print('File name: "{}", Training_Topic {}, Test_topic {}, Score: {}, Label: {}, Found topic and word: {}'.format(test_file_data.iloc[i]['File'],
                  training_head_number,
                  testing_head_number,
                  score, test_file_data.iloc[i]['label'], found_word))
            score_list.append((test_file_data.iloc[i]['File'], training_head_number, testing_head_number, score, test_file_data.iloc[i]['label'], alpha, beta))
            
        print("".join(['*' for m in range(asterisk_len)]))
        print("".join(['*' for m in range(asterisk_len)]))  
        
    
        sys.stdout = sys.__stdout__
        
        # It has testing functions, currently!
        # Score list test is used for the testing purpose!
        return (score_list, buffer, alpha, beta) #score_list_test)
    
    # Tokenizing string in the document
    def tokenization_test(self, testing_file_head, min_ngram, max_ngram):
        wn.ensure_loaded()
        
        testing_dict = {}
        
        # If there is (are) tokenized document(s), then it reads
        # the tokenized document
        if (os.path.isfile(dataset_test + '/' + testing_file_head + tokenized_dataset_suffix)):
            testing_dict[testing_file_head] = pd.read_csv(dataset_test + '/' + testing_file_head + tokenized_dataset_suffix,
                                          encoding='utf-8', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL
                                          ,converters={'Text':ast.literal_eval})
        
        # If there is no tokenized document, then it createds
        # tokenized document.
        else:    
            testing_dict[testing_file_head] = pd.read_csv(dataset_test + '/' + testing_file_head + file_name_df_suffix_csv,
                                              encoding='utf-8', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
            testing_dict[testing_file_head]['Text'] = testing_dict[testing_file_head]['Text'].apply(lambda x: self.cab_tokenizer(x, min_ngram, max_ngram))
            
            file_name = dataset_test + '/' + testing_file_head + tokenized_dataset_suffix
            testing_dict[testing_file_head].to_csv(file_name,
                                  index=False, encoding='utf-8',
                                  quoting=csv.QUOTE_ALL)
        
        return testing_dict
    
    # Tokenization of the string asynchronously
    def asynchronous_tokenization(self, min_ngram, max_ngram):
        files_test = []
        
        for dirpath, dirs, files in os.walk(dataset_test):
            files_test.extend(files)
        
        
        
        test_head = [x[:-len(file_name_df_suffix_csv)] for x in files_test if x.endswith(file_name_df_suffix_csv)]
        
        tokenized_result = None
        with Pool(cpu_count()-1) as p:
            pool_async = p.starmap_async(self.tokenization_test, [[i, min_ngram, max_ngram] for i in test_head])
           
            
            tokenized_result = pool_async.get()
        final_dict = {}
        
        
        for i in tokenized_result:
            final_dict.update(i)
        gc.collect()    
        return final_dict
        
def main():
    
    CLDA_evaluation_screen()
    
if __name__ == "__main__":
    main()