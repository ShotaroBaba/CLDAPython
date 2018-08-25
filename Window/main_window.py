# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:45:47 2018

@author: n9648852
"""
#Reading all necessary packages


import tkinter as tk
import os

import xml.etree.ElementTree as ET
import pandas as pd 
from tkinter.filedialog import askdirectory
import csv
import pickle
import numpy as np
import requests
import itertools

import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
dataset_dir = "../../CLDA_data_training"
dataset_test_dir = "../../CLDA_data_testing"
#Download all necessary nltk download
#components
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#Create main menu            
#Defining the lemmatizer

def define_sw():
    
#    stop_word_path = "../../R8-Dataset/Dataset/R8/stopwords.txt"

#    with open(stop_word_path, "r") as f:
#        stop_words = f.read().splitlines()
    
    return set(stopwords.words('english'))# + stop_words)

def lemmatize(token, tag):
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(tag[0], wordnet.NOUN)

    return lemmatizer.lemmatize(token, tag)

#The tokenizer for the documents
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



#Create vectorise files
#Define the function here
#Generate vector for creating the data
def generate_vector():
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=[1,2],
                           min_df=0.02, max_df=0.98)

#Generate count vectorizer
def vectorize(tf_vectorizer, df):
    #Generate_vector
    #df = df.reindex(columns=['text'])  # reindex on tweet

    tf_matrix = tf_vectorizer.fit_transform(df['Text'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    return tf_matrix, tf_feature_names


#Retrieve the text file from the files
def generate_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield (path + '\\' + file)

#Returns texts in the xml documents 
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

#Read all files in the training dataset


#Read the test files for the test purpose
def read_test_files():
    for_test_purpose_data = pd.DataFrame([], columns=['File', 'Text'])
    training_path = []

    #Creating 
    test_path = "../../R8-Dataset/Dataset/ForTest"
    for dirpath, dirs, files in os.walk(test_path):
        training_path.extend(files)
    
    #Remove the files other than xml files
    training_path = [x for x in training_path if x.endswith('xml')]
    #Remove the path where the 
    #Extract only last directory name
#    for_test_purpose_data = {}
#    training_data_list = []
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        
        #training_data_list.append(path_string)
        #Initialise the list of the strings
        #for_test_purpose_data[path_string] = {}
        
        file = test_path + '/' + path_string 
        tree = ET.parse(file)
        
        #Turn the string into
        root = tree.getroot()
        result = ''
        #Retrieving the text from xml files
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        result = result[:-1] #Removing the space
        
                
        #Initialise 
#        for file in generate_files(path):
            #print("Now reading...")
            #print(open(file, "r").read())

        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']))
    
    return for_test_purpose_data


#Create the test vectors 


#Root class for the application 
class Application():
    
        
    def __init__(self):
        self.root = tk.Tk()
#        self.pack()
        self.start_menu()
        
        #The list of the files for the upload
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
        self.CLDA_model_list = []
        self.retrieve_topic_feature_concept_list()
        
        self.root.mainloop()
        
        
    def start_menu(self):
        
        #Bring all the frame which include all the Listbox frame together
        '''
        ##############################################################################
        ###Main frame
        ##############################################################################
        '''
        self.main_listbox_and_result = tk.Frame(self.root)
        self.main_listbox_and_result.pack()
        
        '''
        #######################################
        ####Folder selection & feature generation part
        #######################################
        '''
        
        #Main frame of the selection
        self.frame_for_folder_selection = tk.Frame(self.main_listbox_and_result)
        self.frame_for_folder_selection.pack(side = tk.LEFT)
        
        self.user_drop_down_select_folder_label = tk.Label(self.frame_for_folder_selection , text = "Folder (Topic)\nSelection")
        self.user_drop_down_select_folder_label.pack()
        
        self.frame_for_drop_down_menu_folder = tk.Frame(self.frame_for_folder_selection)
        self.frame_for_drop_down_menu_folder.pack()
        
        self.scrollbar_drop_down_menu_file = tk.Scrollbar(self.frame_for_drop_down_menu_folder, orient = "vertical")
        self.scrollbar_drop_down_menu_file.pack(side = tk.RIGHT, fill = 'y')
        
        self.user_drop_down_select_folder_list = tk.Listbox(self.frame_for_drop_down_menu_folder, exportselection=0)
        self.user_drop_down_select_folder_list.pack(side = tk.LEFT)
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
        self.folder_selection_result_frame = tk.Frame(self.main_listbox_and_result)
        self.folder_selection_result_frame.pack(side = tk.LEFT, anchor = tk.N)
        
        self.user_drop_down_file_selection_results_label = tk.Label(self.folder_selection_result_frame,
                                                                    text = "Folder (Topic)\nSelection Result")
        
        self.user_drop_down_file_selection_results_label.pack(side = tk.TOP) 
        
        self.user_drop_down_folder_selection_results_frame = tk.Frame(self.folder_selection_result_frame)
        self.user_drop_down_folder_selection_results_frame.pack()
        
        
        self.user_drop_down_folder_selection_results_scroll_bar = \
        tk.Scrollbar(self.user_drop_down_folder_selection_results_frame, orient = "vertical")
        self.user_drop_down_folder_selection_results_scroll_bar.pack(side = tk.RIGHT, fill = 'y')
        
        self.user_drop_down_folder_selection_results_scroll_list = tk.Listbox(self.user_drop_down_folder_selection_results_frame,
                                                                              exportselection = 0)
        self.user_drop_down_folder_selection_results_scroll_list.pack(side = tk.LEFT)
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
        ####Word vector generation part
        ####Already created list part
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
        ####Word vector generation part end
        #######################################
        '''
        
        
        self.user_drop_down_select_folder_buttom = tk.Button(self.root, text = "Select Folder")
        self.user_drop_down_select_folder_buttom.pack()
        self.user_drop_down_select_folder_buttom['command'] = self.select_folder_and_extract_xml
        
        self.user_drop_down_select_folder_button_create_vector = tk.Button(self.root,
                                                                           text = "Create Feature\nVector(s)")
        self.user_drop_down_select_folder_button_create_vector.pack()
        self.user_drop_down_select_folder_button_create_vector['command'] = self.create_feature_vector
        
        self.user_drop_down_select_folder_button_create_concept_prob = tk.Button(self.root,
                                                                           text = "Create Concept\nProb(s)")
        self.user_drop_down_select_folder_button_create_concept_prob.pack()
        self.user_drop_down_select_folder_button_create_concept_prob['command'] = self.create_concept_matrix
        
        self.exit_button = tk.Button(self.root, text = 'quit')
        
        self.exit_button.pack(side = tk.BOTTOM)
        self.exit_button['command'] = self.root.destroy
        
       
        
        
        '''
        #######################################
        ####End the main menu
        #######################################
        '''
    #Select the folder and extract the information
    def select_folder_and_extract_xml(self):

        self.folder_directory = askdirectory()
#        self.folder_name = "C:/Users/n9648852/Desktop/New folder for project/RCV1/Training/Training101"        
        
#        folder_name = os.path.basename("C:/Users/n9648852/Desktop/New folder for project/RCV1/Training/Training101")
        temp_substr = os.path.basename(self.folder_directory)
            
        #If the processed file has already exists, then the process of the
        #topics will stop.
        if any(temp_substr in string for string in self.topic_list):
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, 
                                                                            "Topic already exists") 
            return 
        
        for_test_purpose_data = pd.DataFrame([], columns=['File','Topic', 'Text'])
        self.training_path = []
        
        for dirpath, dirs, files in os.walk(self.folder_directory):
            self.training_path.extend(files)
    
        #Remove the files other than xml files
        self.training_path = [x for x in self.training_path if x.endswith('xml')]
        print(self.training_path)
        topic_name = os.path.basename(self.folder_directory)
        
        for path_to_file in self.training_path:
            path_string = os.path.basename(os.path.normpath(path_to_file)) 
            
            #Extract xml information from the files
            #then it construct the information
            file = self.folder_directory + '/' + path_string 
            tree = ET.parse(file)
            root = tree.getroot()
            result = ''
            for element in root.iter():
                if(element.text != None):
                    result += element.text + ' '
            #Remove the remained data
            result = result[:-1]
            
            name_of_the_file = (os.path.basename((path_string)))
            
            for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(name_of_the_file, 
                                                                               topic_name,
                                                                               result)], 
            columns=['File','Topic', 'Text']))
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
        
            self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, 
                                                                            "{} has been read.".format(name_of_the_file))
        for_test_purpose_data.to_csv(dataset_dir + '/' +
                          topic_name + ".csv",
                          index=False, encoding='utf-8',
                          quoting=csv.QUOTE_ALL)
        #print(for_test_purpose_data)
        self.user_drop_down_folder_selection_results_scroll_list.insert(tk.END, 
                                                                            "File read complete!")
        
        #Adding already created data lists
        self.user_drop_down_select_folder_list.insert(tk.END, 
                                                          topic_name + ".csv")
        
        self.document_data.append(for_test_purpose_data)
        
        self.topic_list.append(topic_name + ".csv")
        
        #Sort the data list after appending csv file
        self.topic_list = sorted(self.topic_list)
        
    #Retrieving topic list
    def retrieve_topic_feature_concept_list(self):
        files_tmp = []
        
        for dirpath, dirs, files in os.walk(dataset_dir):
            files_tmp.extend(files)
#        print(files_tmp)
            #only retrieve the files_tmp which end with .csv
            #Initialise the topic list
        self.topic_list = [x for x in files_tmp if x.endswith('.csv')]
        for i in self.topic_list:
            self.user_drop_down_select_folder_list.insert(tk.END, i)
        #Initialise the features_list
        #Extract feature files from the file lists
        #No need to sort the values as the files are already sorted by names
        self.feature_list = [x for x in files_tmp if x.endswith('_f.pkl')]
        
        for i in self.feature_list:
            self.drop_down_list_word_vector_list.insert(tk.END, i)
        
        self.concept_list = [x for x in files_tmp if x.endswith('_c.pkl')]
        
        for i in self.concept_list:
            self.drop_down_concept_prob_vector_list.insert(tk.END, i)
            pass
        
    #Based on the files_tmp made by the vectors 
    #Depending on what test vector u used 
    #The contents of the vector can be changed 
    def create_feature_vector(self):
        
        files_tmp = []
        
        for dirpath, dirs, files in os.walk(dataset_dir):
            files_tmp.extend(files)
        
        #Extract only the csv files and remove extension!
        files_tmp = [os.path.splitext(x)[0] for x in files_tmp if x.endswith('.csv')]
        
        
        for file_string in files_tmp:
            if any(file_string in substring for substring in self.feature_list):
                #Feature already exists
                print("Feature {} already exists".format(file_string +  '_f.pkl'))
            else:
                #Read csv files 
                datum = pd.read_csv(dataset_dir + '/' +file_string + '.csv', encoding='utf-8', sep=',', 
                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
                #Vectorise the document 
                vect = generate_vector()
                vectorized_data, feature_names = vectorize(vect, datum)
                
                with open(dataset_dir + '/' + file_string + '_f.pkl', "wb") as f:
                    pickle.dump([vectorized_data, feature_names], f)
                
                self.drop_down_concept_prob_vector_list.insert(tk.END, file_string + '_f.pkl')
                self.feature_list.append(file_string + '_f.pkl')
        
        #Sort feature list after appending some elements
        self.feature_list = sorted(self.feature_list)
    
    def retrieve_data(feature_name, K = 10):
        print('Now processing ' + str(feature_name) + " word...")

        #Replace space to + to tolerate the phrases with space
        req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
        + feature_name.replace(' ', '+') + '&topK=' + str(K)
        response = requests.get(req_str)
        
        #Retrieve the response from json file
        return response.json()
    
    
    #Data directory and object should be in arguments to 
    #optimise the data
    def create_concept_matrix(self):
        
        files_tmp = []
        
        #Checking entire files in the text
        for dirpath, dirs, files in os.walk(dataset_dir):
            files_tmp.extend(files) 
        rem_len = -len('_f.pkl')
        #Check whether the concept file(s) exist(s)
        files_tmp = [x[:rem_len] for x in files_tmp if x.endswith('_f.pkl')]
        "ssss"[3:4]
        for file_string in files_tmp:
            #Check whether the test subject exists or no
            if any(file_string in substring for substring in self.concept_list):
                #Feature already exists
                print("Feature {} already exists".format(file_string +  '_c.pkl'))
            else:
                p_e_c  = {}
                j = 0
                
                feature_names = None                    
                with open(dataset_dir + '/' + file_string + '_f.pkl', "rb") as f:
                     _, feature_names = pickle.load(f)
                     
                '''
                #Retrieve the tenth rankings of the words
                
                #K needed to be adjustable so that the
                #Researcher can find the characteristics of
                #all values!
                '''
                    
     
                
#                K = 10 #Retrieve as much as it could
#                for i in feature_names:
#                    print('Now processing ' + str(j) + " word...")
#                    j += 1
#                    #Replace space to + to tolerate the phrases with space
#                    req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
#                    + i.replace(' ', '+') + '&topK=' + str(K)
#                    response = requests.get(req_str)
#                    
#                    #Retrieve the response from json file
#                    p_e_c[i] = response.json()  
#                with open("../../R8-Dataset/Dataset/ForTest/pec_prob_top.pkl", "wb") as f:
#                    pickle.dump(p_e_c, f)
                    
                #List up the concept names
                l = [list(i.keys()) for i in list(p_e_c.values())]
                concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
                
                #    concept_sets[len(concept_sets)-1]
                #Put the atom concept if there are no concepts in the words
                
                
                #Adding atomic elements
                for i in feature_names:
                    #if there are no concepts in the words, then...
                    if p_e_c[i] == {}:
                        
                        #Append the words with no related concpets
                        #as the atomic concepts
                        concept_names.append(i)
                   
                #Sorting the concept_names after adding feature names
                concept_names = sorted(concept_names)
                p_e_c_array= np.zeros(shape=(len(feature_names), len(concept_names)))
                
                #Make it redundant
                #Structure of array
                #p_e_c_array[word][concept]
                for i in p_e_c.keys():
                    for j in p_e_c[i].keys():
                        p_e_c_array[feature_names.index(i)][concept_names.index(j)] = p_e_c[i][j]       
                
                with open(dataset_dir + '/' + file_string +  '_c.pkl', "wb"):
                    pickle.dump([p_e_c_array, concept_names], f)
                    
                self.concept_list.append(file_string +  '_c.pkl')
                self.drop_down_concept_prob_vector_list.insert(tk.END, file_string +  '_c.pkl')
        
        #Sort after adding all values
        self.concept_list = sorted(self.concept_list)
#        for i in self.feature_list:
#            
#            
#            
#        
#        
#        for i in self.topic_list:
#            pass
#            
#        vect = generate_vector()
#        vectorise_data, feature_names = vectorize(vect, test_data)    
#        pass
    
    
def main():
    #Run the main application
    Application()


if __name__ == "__main__":
    main()
