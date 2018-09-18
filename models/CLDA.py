# -*- coding: utf-8 -*-
"""
@author: Shotaro Baba
"""

import pandas as pd
import numpy as np
import os


import requests
import itertools
#import pickle
import sys
#import json
#import csv
#import datetime
#import time
#from multiprocessing import Pool
#import pickle
#import csv
#import json
dataset_dir = "../../CLDA_data_training"
dataset_test = "../../CLDA_data_testing"
score_result_dir = "../../score_result"
score_result_dataframe_suffix = "_score.csv"
score_result_txt_suffix = "_score.txt"
stop_word_folder = "../stopwords"
concept_prob_suffix_json = "_c_prob.json"
concept_name_suffix_txt = "_c_name.txt"
feature_matrix_suffix_csv = "_f_mat.csv"
feature_name_suffix_txt = "_f_name.txt"
file_name_df_suffix_csv = "_data.csv"
CLDA_suffix_pickle = "_CLDA.pkl"
LDA_suffix_pickle = "_LDA.pkl"
converted_xml_suffix = "_conv.txt"
default_score_threshold = 0.10
delim = ","
import asyncio
import concurrent.futures




#import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import CountVectorizer

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
import xml.etree.ElementTree as ET

stop_word_folder = "../stopwords"
stop_word_smart_txt = "smart_stopword.txt"

smart_stopwords = []
with open(stop_word_folder + '/' + stop_word_smart_txt , "r") as f:
    for line in f:
        if not line.startswith('#'):
            #Remove the \n
            smart_stopwords.append(line.strip('\n'))



# initialize constants
lemmatizer = WordNetLemmatizer()

#rand = 11

"""Preparation for the pre-processing"""

# Setting stop words 
def define_sw():
    
    # Use default english stopword     
    return set(stopwords.words('english') + smart_stopwords)


# Defining the lemmatizer
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
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=[1,1],
                           min_df=0.02, max_df=0.98)
    
# Generate document word count vector
# This vector shows how many number of each word is in
# in each document.
def vectorize(tf_vectorizer, df):
    # Generate_vector
    # df = df.reindex(columns=['text'])  #  reindex on tweet

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
def read_test_files(test_path):
    # Construct pandas dataframe
    for_test_purpose_data = pd.DataFrame([], columns=['File', 'Text'])
    training_path = []
    
    # Retrieving the file paths
    for dirpath, dirs, files in os.walk(test_path):
        training_path.extend(files)
    
    # Remove the files other than xml files
    training_path = [x for x in training_path if x.endswith('xml')]
    
    # For each file, a xml file changes to a text file
    # by parsing xml in the file
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        

        file = test_path + '/' + path_string 
        tree = ET.parse(file)
        
        # Turn the string into text file
        root = tree.getroot()
        result = ''
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        # Removing the white space from the result.
        result = result[:-1]
        
        


        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']), ignore_index=True)
    
    return for_test_purpose_data


# Retrieve the data simultaneously
# K represents the range of ranking you want to 
# Retrieve from the Probase database.
def retrieve_p_e_c(feature_names, K = 20, smooth = 0.0001):
#    responses  = {}
#    j = 0
    # Asynchronically fetch p(e|c) data from Microsoft Probase database
    async def retrieve_word_concept_data(feature_names):
            # Setting the max workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=220) as executor:
                collection_of_results = []
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor, 
                        requests.get, 
                        'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' +
                        i.replace(' ', '+') + 
                        '&topK=' + str(K) +
                        '&smooth=' + str(smooth) 
                    )
                    for i in feature_names
                ]
                # Wait the response from the tasks until the data retrieval finishes
                for response in await asyncio.gather(*futures):
                    collection_of_results.append(response.json())
                
                return collection_of_results
    # Retrieve the 
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(retrieve_word_concept_data(feature_names))
    results = loop.run_until_complete(future)    
    # Retrieve the tenth rankings of the words
    p_e_c = {}
    
    for idx, i  in enumerate(feature_names):
#    print(i)
#    print(idx)
        p_e_c[i] = results[int(idx)]
        
    
    return p_e_c


# Now still testing my codes in the
# 
def retrieve_data(feature_name):
        K = 10
        print('Now processing ' + str(feature_name) + " word...")

        # Replace space to + to tolerate the phrases with space
        req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
        + feature_name.replace(' ', '+') + '&topK=' + str(K)
        response = requests.get(req_str)
        
        # Retrieve the response from json file
        return response.json()
#


#Main method for test purpose
def main():
    # Set your folder path here.
    test_path = "../../R8-Dataset/Dataset/ForTest_c"
    test_data = read_test_files(test_path)

    vect = generate_vector()
    vectorised_data, feature_names = vectorize(vect, test_data)        

    
    
    p_e_c = retrieve_p_e_c(feature_names)
    l = [list(i.keys()) for i in list(p_e_c.values())]
    concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
    
#     concept_sets[len(concept_sets)-1]
    
    # Put the atom concept if there are no concepts in the words
    
    file_lists = test_data['File']
    # Adding atomic elements
    for i in feature_names:
        # if there are no concepts in the words, then...
        if p_e_c[i] == {}:
            
            # Append the words with no related concpets
            # as this is atomic concepts by definition
            concept_names.append(i)
    
    
    concept_names = sorted(concept_names)

    
    # Create CLDA object
#    t = LDA( file_lists,feature_names , 5, 20)
    t = CLDA(feature_names, concept_names, file_lists, 5, 20)
    # Run the methods of CLDA to calculate the topic-document, topic-concept probabilities.
    
    vectorised_data= vectorised_data.toarray()
    t.run(vectorised_data, p_e_c)
    # This list is unused but used for testing purpose...
    concept_word_list = t.show_word_concept_prob(p_e_c)
    
#    for i in concept_word_list:
#        print(i)
# The below is done for testing purpose
#    [x for x in t.document_topic_concept_word.values() if (x[3]) == (t.feature_names.index('humiliate'))]
    
#    t.document_topic_concept_word
#    # (m,z,c,w)
#
#    t.show_doc_topic_ranking(2)
#    t.show_concept_topic_ranking(10)
#    t.show_normalized_concept_topic_ranking()
#    
#    # Adjust the number value (Topic) and concept string as
#    # you would like by looking at the normalized_concept_topic
#    # ranking result above.
#    
#    # Usage: show_word_prob_under_concept_topic(topic_number (integer value)
#    # concept_name 'string_value', p(e|c) values dictionary)
#    t.show_word_prob_under_concept_topic(4,'promotional information', p_e_c)
#    
#    with open("CLDA_result.pkl", "wb") as f:
#        pickle.dump(t,f)
#    
#    
#    data_dir = dataset_dir
#    data_test_dir = dataset_test
#    #The topic name (folder name containing the names)
#    topic_name = "Training105"
#    test_name = "Test105"
#    ''.join(filter(str.isdigit, test_name))
#    #########################################################
#    #####This region is for the test of CLDA methods...
#    #########################################################
#    files_training = []
#    files_test = []
#    for dirpath, dirs, files in os.walk(data_dir):
#            files_training.extend(files)
#    
#    for dirpath, dirs, files in os.walk(data_test_dir):
#            files_test.extend(files)
##        print(files_tmp)
#            # only retrieve the files_tmp which end with .csv
#            # Initialise the topic list
##    training_head = [x[:-len(file_name_df_suffix_csv)] for x in files_training if x.endswith(file_name_df_suffix_csv)]
##    test_head = [x[:-len(file_name_df_suffix_csv)] for x in files_test if x.endswith(file_name_df_suffix_csv)]
#    
#    test_file_names = pd.read_csv(data_dir + '/' + topic_name + file_name_df_suffix_csv, encoding='utf-8', sep=',', 
#                            error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
#    
#    len(test_file_names)
#    test_feature_vec, test_feature_names = (None, [])
#        
#    with open(data_dir + '/' + topic_name + feature_name_suffix_txt, "r") as f: 
#        for line in f:
#            #Remove the \n
#            test_feature_names.append(line.strip('\n'))
#    
#    with open(data_dir + '/' + topic_name + feature_matrix_suffix_csv, "r") as f:
#        test_feature_vec = np.loadtxt(f, delimiter = delim)
#    
#    test_concept_prob, test_concept_names = (None, [])
#        
#    with open(data_dir + '/' + topic_name + concept_prob_suffix_json, "r") as f:
#        test_concept_prob = json.load(f)
#        
#    
#    with open(data_dir + '/' + topic_name + concept_name_suffix_txt, "r") as f:
#        for line in f:
#            test_concept_names.append(line.strip('\n'))
#
#    with open(data_dir + '/' + topic_name + CLDA_suffix_pickle, "rb") as f:
#        test_CLDA = pickle.load(f)
#    test_CLDA.set_the_rankings()

#    
#    

#    
#    test_CLDA.show_doc_topic_ranking()
##    'social institution'
#    test_CLDA.show_word_prob_under_concept_topic(0,"non derivative financial instrument", test_concept_prob)
##
##    ##########################################################
###    ##########################################################
###    ##########################################################
#    data_dir = "../../CLDA_data_testing"
#    
#    #The topic name (folder name containing the names)
#    topic_name = "Test104"
##    
#    ##########################################################
#    ##########################################################
#    ##########################################################
#    
#    test_LDA = None
#    
#    
#    test_feature_vec, test_feature_names = (None, [])
#        
#    with open(data_dir + '/' + topic_name + feature_name_suffix_txt, "r") as f: 
#        for line in f:
#            #Remove the \n
#            test_feature_names.append(line.strip('\n'))
#    
#    with open(data_dir + '/' + topic_name + feature_matrix_suffix_csv, "r") as f:
#        test_feature_vec = np.loadtxt(f, delimiter = delim)
#    
#    test_concept_prob, test_concept_names = (None, [])
#        
#    with open(data_dir + '/' + topic_name + concept_prob_suffix_json, "r") as f:
#        test_concept_prob = json.load(f)
#        
#    
#    with open(data_dir + '/' + topic_name + concept_name_suffix_txt, "r") as f:
#        for line in f:
#            test_concept_names.append(line.strip('\n'))
#
#    with open(data_dir + '/' + topic_name + LDA_suffix_pickle, "rb") as f:
#        test_LDA = pickle.load(f)

#    
#    doc_topic = test_LDA.doc_prob_set[0].sum(axis = 0)/test_LDA.doc_prob_set[0].shape[0]
    #########################################################
    #####This region is for the test
    #########################################################
#    list(test_CLDA.total_results)
    
#    for topic in test_CLDA
    
    
class CLDA(object):
    
    def word_indices(self, vec):
 
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx      
    
    # Initialise the objects in a CLDA instance
    def __init__(self, feature_names, concept_names, file_lists, n_topics = 3, maxiter = 15, alpha = 0.1, beta = 0.1):

        
        # Assign feature names
        self.feature_names = feature_names
        
        # Assign concept names
        self.concept_names = concept_names
        
        # The number of topics
        self.n_topics = n_topics
        
        # Alpha value
        self.alpha = alpha
        
        # File list
        self.file_lists = file_lists
        
        
        # Beta value
        self.beta = beta
        
        # Max gibb's sampling iteration value
        self.maxiter = maxiter
        
        # Relationship between word and concept
        self.word_concept = {}
        
#    p_e_c[feature_names[1223]]
#    t.concept_word_relationship[1223]
#    total_results[1223]
    def _initialize(self, matrix, concept_dict):  
     
        #Take the matrix shapes from 
        
        
        n_docs, vocab_size = matrix.shape
        
        n_concepts = len(self.concept_names)

        # Relationships between concepts and words
        # Can be used for optimising the processes
        # Make it [[]] for consistency for calculation
        self.concept_word_relationship = [[[self.concept_names.index(y),
                                            concept_dict[x][y]]
                                            for y in concept_dict[x].keys()] if concept_dict[x] != {} else [[self.concept_names.index(x),
                                                                   1.0]] for x in self.feature_names]
        
#         matrix = matrix.toarray().copy()
        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) # C_D_T Count document topic
        # number of times topic z and word w co-occur
        self.nzc = np.zeros((self.n_topics, n_concepts)) # Topic size * word size
        self.nm = np.zeros(n_docs) #  The number of documents
        self.nz = np.zeros(self.n_topics) # The number of each topic
        self.topics_and_concepts = {} # Topics and concepts
        self.document_topic_concept_word = {}
#        tmp = {}
        for m in range(n_docs):
#            print(m) #Print the document progress
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            #Count the
            for i, w in enumerate(self. word_indices(matrix[m, :])):
                # Randomly put 
                # choose an arbitrary topic as first topic for word i
                # Change this value to if councept_dict[feature_names[w]] != {}
                
                # if the given feature concept probability is zero... then
                if concept_dict[self.feature_names[w]] != {}:
                    z = np.random.randint(self.n_topics) # Randomise the topics
                    c = np.random.randint(n_concepts) # Randomly distribute the concepts per topics 
                    self.nmz[m,z] += 1 # Distribute the count of topic doucment
                    self.nm[m] += 1 # Count the number of occurrences
                    self.nzc[z,c] += 1 # Counts the number of topic concept distribution
                    self.nz[z] += 1 # Distribute the counts of the number of topics
                    self.topics_and_concepts[(m,i)] = (z,c)
                    self.document_topic_concept_word[(m,i)] = (m,z,c,w) # Storing the information about the word
                else:
                    z = np.random.randint(self.n_topics) # Randomise the topics
                    c = self.concept_names.index(self.feature_names[w]) # Randomly distribute the concepts per topics 
                    self.nmz[m,z] += 1 # Distribute the count of topic doucment
                    self.nm[m] += 1 # Count the number of occurrences
                    self.nzc[z,c] += 1 # Counts the number of topic concept distribution
                    self.nz[z] += 1 # Distribute the counts of the number of topics
                    
                    # For future update, they will be joined together....
                    self.topics_and_concepts[(m,i)] = (z,c)
                    self.document_topic_concept_word[(m,i)] = (m,z,c,w)# Storing the information about the word, topic and concepts and its location
                    
    
    
    def _sample_index(self, p_z, nzc):

        
        choice = np.random.multinomial(1,p_z).argmax() # Making the choice throught the 2-dimensional probability array
        concept = int(choice/nzc.shape[0]) # Extract concept array
        topic = choice % nzc.shape[0] # Extract topic index 
        
        # Return topic and concet index
        return topic, concept

    def _conditional_distribution(self, m, w): #Maybe the computation valuables
       
        concept_size = self.nzc.shape[1]
        n_topics = self.nzc.shape[0]
        p_z_stack = np.zeros((self.nzc.shape[1], self.nzc.shape[0]))
        self.p_z_stack = p_z_stack# For testing purpose
        # Count non-zero value in the 
        # Meaning that the words is the atomic concept
            # Calculate only the 
        for concept_num, concept_prob in self.concept_word_relationship[w]:
            left = (self.nzc[:,concept_num] + self.beta) / (self.nz + self.beta * concept_size)  # Corresponding to the left hand side of the equation 
            right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * n_topics)
            p_z_stack[concept_num] = left * right * concept_prob

            # Normalize the values of p_z_stack
            # Reshaping the probability string 
        return (p_z_stack/np.sum(p_z_stack)).reshape((self.nzc.shape[0] * self.nzc.shape[1],))
        # Calculate the atomic topic distribution calculaiton
        # if there are no positive number in the concept array
        # We might need to have the section "Word_Concept: like"

    def run(self, matrix, concept_dict):

#        self.maxiter = maxiter
        # Gibbs sampling program
        self.phi_set = [] # Storing all results Initialisation of all different models
        self.theta_set = [] # Storing all document_topic relation results & initalisation of all other models
        n_docs, vocab_size = matrix.shape
#        matrix = matrix.toarray().copy()
        # Initalise the values
        self._initialize(matrix, concept_dict)
#        matrix = matrix.toarray().copy()
        for it in range(self.maxiter):
            print(it)
            sys.stdout.flush()
            for m in range(n_docs):
                # Asynchronisation can make the progress faster
                # for conducting gibb sampling algorithm
                for i, w in enumerate(self.word_indices(matrix[m, :])):
                    
                    z,c = self.topics_and_concepts[(m,i)] # Extract the topics and concepts from the doucment 
                    self.nmz[m,z] -= 1# Removing the indices from document topic distribution
                    self.nm[m] -= 1 # Removign the indices from total indices in document
                    self.nzc[z,c] -= 1 # Removing the indices from topic concept distribution
                    self.nz[z] -= 1 # Removing the indices from topic distribution
                    
                    # Calculate the probabilities of both topic and concepts
                    p_z = self._conditional_distribution(m, w) # Put the categorical probability on it
                    # If there are topics, then it returns the random topics based on the 
                    # calculated probabilities, otherwise it returns the atomic bomb
                    z,c = self._sample_index(p_z, self.nzc) # Re-distribute concept and topic 
                    
                    # Randomly adding the indices based on the calculated probabilities
                    self.nmz[m,z] += 1
                    self.nm[m] += 1 # Adding the entity based on the percentage
                    self.nzc[z,c] += 1 # Addign the entity for 
                    self.nz[z] += 1 #  Count the number of the topic occurrences
                    self.topics_and_concepts[(m,i)] = z,c # Re-assign the concepts and topics
                    self.document_topic_concept_word[(m,i)] = (m,z,c,w)
#                     self.document_topic_concept_word[i][2] = c
                    
            # Print the time of the iteration                    
            print("Iteration: {}".format(it))
            # Print phi value(s)
            print("Phi: {}".format(self.phi()))
            # Print the document probability value
            print("Doc_prob: {}".format(self.doc_prob()))
            # Print theta value
            print("Theta: {}".format(self.theta()))
        
        #Storing newest phi value and theta value for calculating word, concept and topic ranking
        self.phi_set.append(self.phi())
        self.theta_set.append(self.doc_prob())
        
        self.set_the_rankings()
#    
    # Calculate phi value
    def phi(self):

        # Not necessary values for the calculation
        # V = nzw.shape[1]
        num = self.nzc + self.beta # Calculate the counts of the number, the beta is the adjust ment value for the calcluation
        num /= np.sum(num, axis=1)[:, np.newaxis] # Summation of all value and then, but weight should be calculated in this case....
        return num
    
    # This is actually theta value...
    def doc_prob(self):
        
#        T = nmz.shape[0]
        num = self.nmz + self.alpha #Cal
        num /= np.sum(num, axis=1)[:, np.newaxis]
        
        return num
    
    # Calculate theta
    def theta(self):
        num = self.nmz.sum(axis = 0) + self.alpha
        num /= np.sum(num)
        
        return num
        
    # Set the word ranking as it takes too much time to
    def set_the_rankings(self):
        
        self.concept_ranking = []
        self.doc_ranking = []
        
#        if(not (self.phi_set in locals() or self.phi.set in globals())):
#            print("The calculation of phi or theta is not done yet!")
        
        # Calcualte the topic_word distribution by sorti
        temp = np.argsort(-(self.phi_set[0]))
        
        # Calculate the topic_document distribuiton
        temp2 = np.argsort(-(self.theta_set[0].T))
        
        # Create the concept ranking
        self.concept_ranking = [[[topic, ranking, self.concept_names[idx], self.phi_set[0][topic][idx], idx] for ranking, idx in enumerate(concept_idx)]
                                for topic, concept_idx in enumerate(temp)]
        
        # Create the document ranking
        self.doc_ranking = [[[topic, ranking, self.file_lists[doc_idx], self.theta_set[0].T[topic][doc_idx]] for ranking, doc_idx in enumerate(docs_idx)]
                    for topic, docs_idx in enumerate(temp2)]
        
        

    
    def show_doc_topic_ranking(self, rank=10):
        
        #Print document probabilities over topics
        print('\n')
        print("*********************************")
        print("Theta value: ")
        print("*********************************")
        print(self.theta())
        print('\n')
        #Each document ranking over topic is printed 
        for i in range(self.nzc.shape[0]):
            print('\n')
            print("#############################")
            print("Topic {} doc prob ranking: ".format(i))
            
          
            rank = min(self.theta_set[0].shape[0], rank)
            for j in range(rank):
                 print('Rank: {}, Doc_name: "{}", Doc_prob value: {}'.format(self.doc_ranking[i][j][1],
                       self.doc_ranking[i][j][2], self.doc_ranking[i][j][3]))
        
        
    def show_concept_topic_ranking(self, rank=10):
        
        
        print('\n')
        print("*********************************")
        print("Phi value: ")
        print("*********************************")
        print(self.phi())
        print('\n')          
        #Each concept ranking over topic is printed
        for i in range(self.nzc.shape[0]):
            print('\n')
            print("#############################")
                    
            print("Topic {} concpet prob ranking: ".format(i))
            
         
            rank = min(self.phi_set[0].shape[1], rank)
            for j in range(rank):
                print('Rank: {}, Concept: "{}", Concept_prob value: {}'.format(self.concept_ranking[i][j][1], 
                      self.concept_ranking[i][j][2], self.concept_ranking[i][j][3]))
    
    
    def show_and_construct_normalized_concept_topic_ranking(self, rank = 10):
        print('\n')
        print("*********************************")
        print("Phi value: ")
        print("*********************************")
        print(self.phi())
        print('\n')          
    #Each concept ranking over topic is printed
        topic_concept_prob = []
        for i in range(self.nzc.shape[0]):
            concept_prob = []
            print('\n')
            print("#############################")
                    
            print("Topic {} concpet prob ranking: ".format(i))
            value_for_normalisation = sum([self.concept_ranking[i][k][3] for k in range(rank)])
            print("#############################")
            print("Value for normalizing rank of Top {}: {}".format(rank, value_for_normalisation))
            
            rank = min(self.phi_set[0].shape[1], rank)
                        
            for j in range(rank):
                print('Rank: {}, Concept: "{}", Concept_prob value: {}'.format(self.concept_ranking[i][j][1],
                      self.concept_ranking[i][j][2], self.concept_ranking[i][j][3]/value_for_normalisation))
                concept_prob.append((self.concept_ranking[i][j][2], self.concept_ranking[i][j][3]/value_for_normalisation))                
            
            topic_concept_prob.append(concept_prob)
        return topic_concept_prob
            
    # Show word(s) under one concept
    def show_word_prob_under_concept_topic(self, topic, concept = None, word_under_concept_probability = None, rank = 10):
            
            if word_under_concept_probability == None:
                print("Please select the values")
            
            concept_index  = self.concept_names.index(concept)
            tmp = list(set(list(self.document_topic_concept_word.values()))) 
            set_of_candidates = list(set(list([(x[1],x[2],x[3],word_under_concept_probability[self.feature_names[x[3]]][self.concept_names[concept_index]]) if
                                          word_under_concept_probability[self.feature_names[x[3]]] != {} else
                                          # If the value is atomic value, then it is regarded as 1
                                          (x[1],x[2],x[3], 1.0) for x in tmp if (topic, concept_index) == (x[1], x[2])])))
            set_of_candidates = sorted(set_of_candidates, key = (lambda x: x[3]), reverse = True)
            
            rank = min(len(set_of_candidates), rank)
            for candidate in set_of_candidates[:rank]:
                out_str = 'topic: {}, concept: {}, word: "{}", probability: {}'.format(topic, self.concept_names[concept_index], self.feature_names[candidate[2]],
                                 candidate[3])
                print(out_str)  
    
    # Show the word under concept(s)
    def construct_word_concept_prob_under_concept(self, word_under_concept_probability):
        
        concept_word_list = []
        concept_word = sorted(set([(self.feature_names[x[3]], self.concept_names[x[2]], 
          word_under_concept_probability[self.feature_names[x[3]]][ self.concept_names[x[2]]]) if 
            word_under_concept_probability[self.feature_names[x[3]]] != {} else 
            (self.feature_names[x[3]], self.concept_names[x[2]], 1.0) for x 
          in list(set(self.document_topic_concept_word.values()))]), key = (lambda x: x[2]), reverse = True)
        
        for word, concept, probability in concept_word:
            concept_word_list.append((word, concept, probability)) 
        

        return concept_word_list
    
    # Done for reduce the object size...
    def show_word_concept_prob(self, word_under_concept_probability):
        concept_word_list = []
        concept_word = sorted(set([(self.feature_names[x[3]], self.concept_names[x[2]], 
          word_under_concept_probability[self.feature_names[x[3]]][ self.concept_names[x[2]]]) if 
            word_under_concept_probability[self.feature_names[x[3]]] != {} else 
            (self.feature_names[x[3]], self.concept_names[x[2]], 1.0) for x 
          in list(set(self.document_topic_concept_word.values()))]), key = (lambda x: x[2]), reverse = True)
        
        for word, concept, probability in concept_word:
            concept_word_list.append((word, concept, probability)) 
        
        concepts = sorted(list(set([x[1] for x in concept_word_list])))
        
        for concept in concepts:
            print("".join(['*' for x in range(20)]))
            print('Concept "{}":'.format(concept))
            for word, concept, probability in [x for x in concept_word_list if x[1] == concept]:
                print('\tWord "{}", Probability: {}'.format(word, probability))
            print("".join(['*' for x in range(20)]))
        
        return concept_word_list    
            
# Baseline method
class LDA(object):

    def __init__(self, file_list, feature_names, n_topics,alpha=0.1, beta=0.1):
       
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.feature_names = feature_names
        self.file_list = file_list
        
    def sample_index(self,p):
        
        return np.random.multinomial(1,p).argmax()
    
    def word_indices(self, vec):
       
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx
    
    def _initialize(self, matrix):
        
        # For test purpose only!
        n_docs, vocab_size = matrix.shape
#        matrix = matrix.toarray().copy()
        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) # C_D_T Count document topic
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size)) # Topic size * word size
        self.nm = np.zeros(n_docs) # Number of documents
        self.nz = np.zeros(self.n_topics) # Number of topic
        self.topics = {} # Topics dictionary

        for m in range(n_docs):
#            print(m)
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(self.word_indices(matrix[m, :])):
                
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics) # Randomise the topics
                self.nmz[m,z] += 1 # Distribute the count of topic doucment
                self.nm[m] += 1 # Count the number of occurrences
                self.nzw[z,w] += 1 # Counts the number of topic word distribution
                self.nz[z] += 1 # Distribute the counts of the number of topics
                self.topics[(m,i)] = z # Memorise the correspondence between topics and the entities
        sys.stdout.flush()
        
    def _conditional_distribution(self, m, w): # Maybe the computation valuables
       
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size) # Corresponding to the left hand side of the equation 
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics) #Corresponding to the right hand side of the equation
        # We might need to have the section "Word_Concept: like"
        # p_e_c = some expression
        p_z = left * right #* P(e|c)
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z



    def phi(self):
       
        # Not necessary values for the calculation
        # V = self.nzw.shape[1]
        num = self.nzw + self.beta # Calculate the counts of the number, the beta is the adjust ment value for the calcluation
        num /= np.sum(num, axis=1)[:, np.newaxis] # Summation of all value and then, but weight should be calculated in this case....
        return num
    
    def doc_prob(self):
        
        # Calculate document probability
        num = self.nmz + self.alpha 
        num /= np.sum(num, axis=1)[:, np.newaxis] 
        
        return num
    
    # Calculate theta
    def theta(self):
        num = self.nmz.sum(axis = 0) + self.alpha
        num /= np.sum(num)
        
        return num
    
    def run(self, matrix, maxiter=30):
       
        # Gibbs sampling
        self.phi_set = [] # Storing all results Initialisation of all different models
        self.theta_set = [] # Storing all document_topic relation results & initalisation of all other model
        self.doc_prob_set = []
        
        n_docs, vocab_size = matrix.shape
#        matrix = matrix.toarray().copy()
        self._initialize(matrix)
        
        for it in range(maxiter):
            print(it)
            sys.stdout.flush()
            for m in range(n_docs):
                
                for i, w in enumerate(self.word_indices(matrix[m, :])):
                    
                    z = self.topics[(m,i)] # The entities of topics, the value c needs to be included in here
                    self.nmz[m,z] -= 1# Removing the indices 
                    self.nm[m] -= 1 # Removign the indices
                    self.nzw[z,w] -= 1 # Removing the indices
                    self.nz[z] -= 1 # Removing the indices

                    p_z = self._conditional_distribution(m, w) # Put the categorical probability on it
                    z = self.sample_index(p_z)

                    self.nmz[m,z] += 1 # Randomly adding the indices based on the calculated probabilities
                    self.nm[m] += 1 # Adding the entity based on the percentage
                    self.nzw[z,w] += 1 # Addign the entity for 
                    self.nz[z] += 1 # Count the number of the occureences
                    self.topics[(m,i)] = z # Re=assignm the topic

            # Retrieve the phi, theta and document topic co-occurrence probability
            # values
            print('\n' + "iteration: {}".format(it) + '\n')
            print("phi: {}".format(self.phi()))
            print("Theta: {}".format(self.theta()))
            print("doc_prob: {}".format(self.doc_prob()))
            
        self.phi_set.append(self.phi())
        
        # Document Probability!
        self.doc_prob_set.append(self.doc_prob())
        
        self.maxiter = maxiter
        
        # Ranking is automatically set
        # after a series of process
        self.set_the_rankings()
    
    # Testing the programs for displaying the data
    # Testing
    
    def set_the_rankings(self):
        
        self.word_ranking = []
        self.doc_ranking = []
        
#         if(not (self.phi_set in locals() or self.phi.set in globals())):
#             print("The calculation of phi or theta is not done yet!")
            
        # Calcualte the topic_word distribution
        temp = np.argsort(-(self.phi_set[0]))
        
        # Calculate the topic_document distribuiton
        temp2 = np.argsort(-(self.doc_prob_set[0].T))
        
        # Create the leaderships 
        self.word_ranking = [[[topic, ranking, self.feature_names[idx], self.phi_set[0][topic][idx]] for ranking, idx in enumerate(word_idx)]
                                for topic, word_idx in enumerate(temp)]
        
        self.doc_ranking = [[[topic, ranking, self.file_list[doc_idx], self.doc_prob_set[0].T[topic][doc_idx]] for ranking, doc_idx in enumerate(docs_idx)]
                    for topic, docs_idx in enumerate(temp2)]
    
    def generate_word_prob(self, rank = 10):
        word_topic_list = []
        # In this rank, all rank is normalised
#        rank = min(self.phi_set[0].shape[1], rank)
        for x in self.word_ranking:
            rank = min(len(x), rank)
            rank_for_normalization = sum(list(zip(*x))[3][:rank])
            word_topic_list.extend([(j[0], j[2], j[3]/rank_for_normalization) for j in x[:rank]])
            
#            sum(list(zip(*word_topic_list))[2])
        return word_topic_list
        
    # Show the document ranking over topic
    def show_doc_topic_ranking(self, rank=10):

        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            
            rank = min(self.doc_prob_set[0].shape[0], rank)
            for j in range(rank):
                 print('Rank: {}, Document: "{}", Probability: {}'.format(self.doc_ranking[i][j][1], self.doc_ranking[i][j][2], self.doc_ranking[i][j][3]))
        
    # Show the topic ranking over the words
    def show_word_topic_ranking(self, rank=10):
        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            
#            test_LDA.phi_set[0].shape[1]
#            test_LDA.doc_prob_set[0].shape[0]
            rank = min(self.phi_set[0].shape[1], rank)
            for j in range(rank):
                print('Rank: {}, Word: "{}", Probability: {}'.format(self.word_ranking[i][j][1], self.word_ranking[i][j][2], self.word_ranking[i][j][3]))
            
#    def construct_word_list_with_ranking(self, rank = 10):
#        for i in range(self.nzw.shape[0]):
#            
#            rank = min(self.phi_set[0].shape[1], rank)
#            for j in range(rank):
#                
            

##
if (__name__ == "__main__"):
#    pass
    main()
