# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:10:53 2018

@author: n9648852
"""

import pandas as pd
import numpy as np
import os
import csv
import pickle
import urllib
import requests
import itertools

from sklearn.decomposition import LatentDirichletAllocation

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
#from sklearn.feature_extraction.text import CountVectorizer

import xml.etree.ElementTree as ET
#from pyspark.mllib.fpm import FPGrowth
#
#import pyspark

# importing some libraries

#from pyspark.sql import SQLContext

# stuff we'll need for text processing
import requests
import re as re



# initialize constants
lemmatizer = WordNetLemmatizer()

#Set the random state number

rand = 11

"""Preparation for the pre-processing"""

#Setting stop words 
def define_sw():
    
    stop_word_path = "../../R8-Dataset/Dataset/R8/stopwords.txt"

    with open(stop_word_path, "r") as f:
        stop_words = f.read().splitlines()
    
    return set(stopwords.words('english') + stop_words)

#def create_lda(tf_matrix, params):
#    return LatentDirichletAllocation(n_components=params['lda_topics'], max_iter=params['iterations'],
#                                     learning_method='online', learning_offset=10,
#                                     random_state=0).fit(tf_matrix)


#Defining the lemmatizer
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
                           min_df=0.02, max_df=0.98, inputCol="words", outputCol="vectors")

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
def read_R8_files(path):
    
    #Setting the training path for reading xml files
    RCV1_training_path = path
    path = RCV1_training_path
    #RCV1_training_path = "../../R8-Dataset/Dataset/R8/Training"
    #RCV1_training_path = "../../R8-Dataset/Dataset/R8/Testing"
    #RCV1_test_path = "../../R8-Dataset/Dataset/R8/Testing"
    #os.path.dirname(RCV1_training_path)
    
    training_path = []
    
    for dirpath, dirs, files in os.walk(RCV1_training_path):
        training_path.append(dirpath)
    
    
    #Remove the directory itself so that there are no troubles
    training_path.remove(RCV1_training_path)
    
    training_text_data = pd.DataFrame([], columns=['Topic', 'File', 'IsTopic', 'Text'])
    
    #Extract only last directory name
#    training_text_data = {}
#    training_data_list = []
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        
        #training_data_list.append(path_string)
        #Initialise the list of the strings
        #training_text_data[path_string] = {}
        print("Now reading...")
        print(path_string)
        #Initialise 
#        for file in generate_files(path):
            #print("Now reading...")
            #print(open(file, "r").read())
        df = pd.read_csv("../../R8-Dataset/Dataset/R8/Topic/" + path_string + ".txt", 
                              sep = '\s+',  names = ['Topic', 'File', 'IsTopic'])
        df['Text'] = None
        df = df[df.IsTopic == 1]
        df['Text'] = df.File.apply(lambda row: return_text(path, path_string, row))
        training_text_data.info()
        training_text_data = training_text_data.append(df)
        len(training_text_data)
#            if(not (file.endswith(".txt"))):
#                tree = ET.parse(file)
#                root = tree.getroot()
#                result = ''
#                for element in root.iter():
#                    if(element.text != None):
#                        result += element.text + ' '
#                result = result[:-1]
#                
#                with open(file + ".txt", "w", encoding = 'utf-8') as f:
#                    f.write(result)
#                    
#                
#                #Obtaining all the text data from the document 
#                training_text_data = \
#                training_text_data.append(pd.DataFrame([[path_string, result]], columns=['Frame', 'Text']))
                
                
                
            #training_text_data[path_string].append(result)
    
    #Write the data into the file
    #training_text_data[path_string][os.path.splitext(os.path.basename(file))[0]]
    training_text_data.to_csv("../../R8-Dataset/" +
                              os.path.basename(os.path.normpath(RCV1_training_path))+ ".csv",
                              index=False, encoding='utf-8',
                              quoting=csv.QUOTE_ALL)
#    df = pd.DataFrame.from_dict(training_text_data)
    #df['Training101']
    
    return training_text_data
    #Now reading the test data
    #Showing what data is read by the readers


if(not os.path.isfile("../../R8-Dataset/Training.csv")):
    R8_training_data = read_R8_files("../../R8-Dataset/Dataset/R8/Training")
if(not os.path.isfile("../../R8-Dataset/Testing.csv")):
    R8_test_data = read_R8_files("../../R8-Dataset/Dataset/R8/Testing")

R8_training_data = pd.read_csv("../../R8-Dataset/Training.csv" , encoding='utf-8', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
R8_test_data = pd.read_csv("../../R8-Dataset/Testing.csv", encoding='latin1', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)

R8_training_data.info()

##################Frequent mining R8 (in progress)##############################
################################################################################
################################################################################
################################################################################

R8_training_data = pd.read_csv("../../R8-Dataset/Training.csv" , encoding='utf-8', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
R8_test_data = pd.read_csv("../../R8-Dataset/Testing.csv", encoding='latin1', sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)
df_concatenated_text = R8_training_data.groupby('Topic').Text.apply(lambda x: ' '.join(x)).reset_index()

#Running spark if it does not run, or if it exists

    

    
    #result = model.freqItemsets().collect()


#    for fi in result:
#        print(fi)


#df_concatenated_text = R8_training_data.groupby('Topic').Text.apply(lambda x: ' '.join(x)).reset_index()
#df_concatenated_text['list'] = df_concatenated_text.Text.apply(lambda x: list(set(x.split())))
#df_concatenated_text['model'] = df_concatenated_text.list.apply(lambda x: FPGrowth.train(sc.parallelize(x, 2), 0.6, 2))
#model = FPGrowth.train(sc.parallelize(df_concatenated_text.list, 2), 0.6, 2)
#
#sorted(model.freqItemsets().collect())


#Reading RCV1 files (in progress)###############################################
################################################################################
################################################################################
################################################################################

##Topic modelling for generating the models 

#Read all files in the training dataset
def read_RCV1_files(path):
    #Setting the training path for reading xml files
    #RCV1_training_path = "../../RCV1-20180729T045619Z-001/RCV1/Training"
    RCV1_training_path = path
    #RCV1_test_path = "../../R8-Dataset/Dataset/R8/Testing"
    #os.path.dirname(RCV1_training_path)
    
    training_path = []
    
    for dirpath, dirs, files in os.walk(RCV1_training_path):
        training_path.append(dirpath)
    
    
    
    training_path.remove(RCV1_training_path)
    
    #Extract only last directory name
    training_text_data = {}
    training_data_list = []
    
    for path in training_path:
        path_string = os.path.basename(os.path.normpath(path))
        
        training_data_list.append(path_string)
        #Initialise the list of the strings
        
        
        #Initialise 
        if(len(list(generate_files(path))) != 0):
            training_text_data[path_string] = {}
            #path =  '../../RCV1-20180729T045619Z-001/RCV1/Training\\Training200'
            for file in generate_files(path):
                #print("Now reading...")
                print(open(file, "r").read())
                tree = ET.parse(file)
                root = tree.getroot()
                result = ''
                for element in root.iter():
                    if(element.text != None):
                        result += element.text + ' '
                result = result[:-1]
                
                #Obtaining all the text data from the document 
                training_text_data[path_string][os.path.splitext(os.path.basename(file))[0]] = result
                #training_text_data[path_string].append(result)
    
    
    #training_text_data[path_string][os.path.splitext(os.path.basename(file))[0]]
#    training_text_data['Training199']
    df = pd.DataFrame.from_dict(training_text_data)
#    #df['Training101']
    df.info()
    return df



    #df.head()
    #Now reading the test data
    #Showing what data is read by the readers
#RCV1_training_data = read_RCV1_files("../../RCV1-20180729T045619Z-001/RCV1/Training")
#RCV1_test_data = read_RCV1_files("../../RCV1-20180729T045619Z-001/RCV1/Testing")
#RCV1_training_data['Training101'][2]
#RCV1_training_data.info()
#Create the training data
def frequent_closed_pattern_mining(training):
    #Retrieve data from the 
    return
#return
    
    #for path in training_path
    

######################################################################
######################################################################
######################################################################


###Commented out the following codes#####################################
#########################################################################
#########################################################################

#R8_training_data.info()
#R8_test_tf_matrix, R8_test_tf_feature_names = vectorize(generate_vector(), R8_test_data)

#Saving the object file for shortening the computation tieme


#R8_training_tf_matrix, R8_training_tf_feature_names = vectorize(generate_vector(), R8_training_data)

training_doc_name = R8_training_data['File']
testing_doc_name = R8_test_data['File']

training_cat = R8_training_data['Topic']
testing_cat = R8_test_data['Topic']

R8_training_tf_matrix, R8_training_tf_feature_names = (None, None)

R8_training_data = pd.read_csv("../../R8-Dataset/Training.csv" , encoding='utf-8', 
                               sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)


training_doc_name = R8_training_data['File']
testing_doc_name = R8_test_data['File']

training_cat = R8_training_data['Topic']
testing_cat = R8_test_data['Topic']

#Generate the vectorised data if it doesn't exist
if(not os.path.isfile("../../R8-Dataset/Vectorised_training.pkl")):
    R8_training_tf_matrix, R8_training_tf_feature_names = vectorize(generate_vector(), R8_training_data)
    with open("../../R8-Dataset/Vectorised_training.pkl", "wb") as f:
        pickle.dump([R8_training_tf_matrix, R8_training_tf_feature_names], f)

if(not os.path.isfile("../../R8-Dataset/Vectroised_training_docname.pkl")):
    with open("../../R8-Dataset/Vectroised_training_docname.pkl", "wb") as f:
        pickle.dump(training_doc_name, f)
        


if(not os.path.isfile("../../R8-Dataset/Vectroised_testing_docname.pkl")):
    with open("../../R8-Dataset/Vectroised_testing_docname.pkl", "wb") as f:
        pickle.dump(testing_doc_name, f)

if(not os.path.isfile("../../R8-Dataset/Vectroised_training_catname.pkl")):
    with open("../../R8-Dataset/Vectroised_training_catname.pkl", "wb") as f:
        pickle.dump(training_cat, f)

if(not os.path.isfile("../../R8-Dataset/Vectroised_testing_catname.pkl")):
    with open("../../R8-Dataset/Vectroised_testing_docname.pkl", "wb") as f:
        pickle.dump(testing_cat, f)

def create_lda(tf_matrix):
    return LatentDirichletAllocation(n_components=5, max_iter=5,
                                     learning_method='online', learning_offset=10,
                                     random_state=rand).fit(tf_matrix)





def create_tw_dist(model):
    # return normalized topic-word distribution
    normTWDist = model.components_ / \
        model.components_.sum(axis=1)[:, np.newaxis]

    return normTWDist

def create_dt_dist(model, tf_matrix):
    # return normalized document-topic distribution
    normDTDist = model.transform(tf_matrix)

    return normDTDist

R8_dirichlet_allocation = create_lda(R8_training_tf_matrix)
#R8_training_tf_matrix
R8_document_topic_distribution = create_dt_dist(R8_dirichlet_allocation, R8_training_tf_matrix) 


def create_topic_dataframe(dt_dist):
    
    #Create the list for generating topics 
    topics = []
    for i in range(dt_dist.shape[1]):
        topics.append("topic_" + str(i))
    
    #Return the topic distribution of each document
    return pd.DataFrame(dt_dist, columns = topics)
topic_words = {}
vocab = R8_training_tf_feature_names
df_R8_dt = create_topic_dataframe(R8_document_topic_distribution)        
    
df_with_dt = pd.concat([R8_training_data, df_R8_dt], axis = 1)

for topic, comp in enumerate(R8_dirichlet_allocation.components_ / 
                             R8_dirichlet_allocation.components_.sum(axis=1)[:, np.newaxis]):
    
    word_idx = np.argsort(comp)[::-1][:]
    topic_words[topic] = [[vocab[i],comp[i]] for i in word_idx]

for topic, words in topic_words.items():
    print('Topic: %d' % topic)
    print(words[0:20])

##############################################################################
##############################################################################
##############################################################################
##############################################################################
    
###########Concept word distribution##########################################
##############################################################################
##############################################################################
##############################################################################

#Conceptualised latent dirichlet allocation

##P(e|c) score
#response = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=microsoft&topK=10')
##P(c|e) score
#response.json()
#
#response = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=moe&topK=20')
#response.json()

len(R8_training_tf_feature_names)

def replace_space_with_plus(string):
    return re.compile()

#Calculate all P(c|e) values
def retrieve_p_e_c(feature_names):
    feature_names = R8_training_tf_feature_names
    responses  = {}
    j = 0
    if(not os.path.isfile("../../R8-Dataset/pec_prob_top.pkl")):
        K = 20 #Retrieve as much as it could
        for i in feature_names:
            print('Now processing ' + str(j) + " word...")
            j += 1
            #Replace space to + to tolerate the phrases with space
            req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
            + i.replace(' ', '+') + '&topK=' + str(K)
            response = requests.get(req_str)
            
            responses[i] = response.json()  
        with open("../../R8-Dataset/pec_prob_top.pkl", "wb") as f:
            pickle.dump([responses], f)
        return responses
    else:
        print('File exist... stop the program...')






#def retrieve_p_c_e(feature_names):
#    feature_names = R8_training_tf_feature_names
#    responses  = {}
#    j = 0
#    if(not os.path.isfile("../../R8-Dataset/pce_prob_top.pkl")):
#        K = 20 #Retrieve as much as it could
#        for i in feature_names:
#            print('Now processing ' + str(j) + " word...")
#            j += 1
#            #Replace space to + to tolerate the phrases with space
#            req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=' \
#            + i.replace(' ', '+') + '&topK=' + str(K)
#            response = requests.get(req_str)
#            
#            responses[i] = response.json()  
#        with open("../../R8-Dataset/pce_prob_top.pkl", "wb") as f:
#            pickle.dump([responses], f)
#        return responses
#    else:
#        print('File exist... stop the program...')
#p_c_e = None
##p_c_e = retrieve_p_c_e(R8_training_tf_feature_names)
#
#with open("../../R8-Dataset/pce_prob_top.pkl", "rb") as f:
#    p_c_e = pickle.load(f)[0]   
#
#l = [list(i.keys()) for i in list(p_c_e.values())]
#concept_sets = sorted(list(set(itertools.chain.from_iterable(l))))
#

R8_training_tf_matrix, R8_training_tf_feature_names,  = (None, None)

R8_training_data = pd.read_csv("../../R8-Dataset/Training.csv" , encoding='utf-8', 
                               sep=',', error_bad_lines = False, quotechar="\"",quoting=csv.QUOTE_ALL)


training_doc_name = R8_training_data['File']
#testing_doc_name = R8_test_data['File']

training_cat = R8_training_data['Topic']
#testing_cat = R8_test_data['Topic']

#Generate the vectorised data if it doesn't exist
if(not os.path.isfile("../../R8-Dataset/Vectorised_training.pkl")):
    R8_training_tf_matrix, R8_training_tf_feature_names = vectorize(generate_vector(), R8_training_data)
    with open("../../R8-Dataset/Vectorised_training.pkl", "wb") as f:
        pickle.dump([R8_training_tf_matrix, R8_training_tf_feature_names], f)

#Load the file containing the variables
with open("../../R8-Dataset/Vectorised_training.pkl", "rb") as f:   
    R8_training_tf_matrix, R8_training_tf_feature_names = pickle.load(f)
    
#with open("../../R8-Dataset/Vectroised_training_docname.pkl", "rb") as f:
#    training_doc_name = pickle.load(f)
#
##Load the file containing the variables
#with open("../../R8-Dataset/Vectroised_testing_docname.pkl", "rb") as f:   
#    testing_doc_name = pickle.load(f)




#p_e_c = None
#with open("../../R8-Dataset/pec_prob_top.pkl", "rb") as f:
#    p_e_c = pickle.load(f)[0]   
#i = None
#
#l = [list(i.keys()) for i in list(p_e_c.values())]
#concept_sets = sorted(list(set(itertools.chain.from_iterable(l))))
#
#R8_training_tf_feature_names
##p_e_c_array[0])
#
#p_e_c_array= np.zeros(shape=(len(R8_training_tf_feature_names), len(concept_sets)))
#
#R8_training_tf_feature_names[0]
##Retrieve all possible concepts from the string
#
##ce_responces = retrieve_p_c_e(R8_training_tf_feature_names)
#len(concept_sets)
#
#concept_sets[3]

#retrieve_p_e_c(R8_training_tf_feature_names)


#p_e_c['0']['1-digit activity code']
#
##Insert the numbers corresponding to the arrays
#for i in p_e_c.keys():
#    for j in p_e_c[i].keys():
#        p_e_c_array[R8_training_tf_feature_names.index(i)][concept_sets.index(j)] = p_e_c[i][j]
#
#
#
#if(not os.path.isfile("../../R8-Dataset/p_e_c_array_top.pkl")):
#    with open("../../R8-Dataset/p_e_c_array_top.pkl", "wb") as f:
#       pickle.dump(p_e_c_array, f) 

import numpy as np
import scipy as sp
from scipy.special import gammaln

R8_training_tf_matrix




#Assuming that the topic number is 5





def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


#def CLDA(n_topics, training_names, training, alpha=0.1, beta=0.1):
#    
#    
#    def initialize(matrix):
#        
#        
#        return 
#    
#    
#    return

#def word_indices(matrix):
#    """
#    Turn a document vector of size vocab_size to a sequence
#    of word indices. The word indices are between 0 and
#    vocab_size-1. The sequence length is equal to the document length.
#    """
#    
#    vec = R8_training_tf_matrix.toarray()
#    w_indices = []
##   Returning the index numbers   
##    for idx in vec.nonzero()[0]:
##        for i in range(int(vec[idx])):
##            yield idx
#    
#    for idx in vec.nonzero()[0]:
#        w_indices.extend([vec[0].nonzero() for iteration in range(vec[idx])])
#    return  w_indices
#
#R8_training_tf_matrix.nonzero()
#t = np.ndenumerate(R8_training_tf_matrix.nonzero()[0].shape)
#next(t)
#for x, i in np.ndenumerate(R8_training_tf_matrix.nonzero()):
#    if(x[0] != 0):
#        print(x)
#        print(i)
#        if i == 3:
#            break
#def sample_index(p):
#    """
#    Sample from the Multinomial distribution and return the sample index.
#    """
#    return np.random.multinomial(1,p).argmax()
#
#    
#
#
#def word_indices(vec):
#    """
#    Turn a document vector of size vocab_size to a sequence
#    of word indices. The word indices are between 0 and
#    vocab_size-1. The sequence length is equal to the document length.
#    """
#    for idx in vec.nonzero()[0]:
#        for i in range(int(vec[idx])):
#            yield idx

class LDA(object):

    def __init__(self, n_topics,alpha=0.1, beta=0.1):
        """
        n_topics: number of topics
        
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
    
    def sample_index(self,p):
        """
        Sample from the Multinomial distribution and return the sample index.
        """
        return np.random.multinomial(1,p).argmax()
    
    def word_indices(self, vec):
        """
        Turn a document vector of size vocab_size to a sequence
        of word indices. The word indices are between 0 and
        vocab_size-1. The sequence length is equal to the document length.
        """
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx
    
    def _initialize(self, matrix):
        
        #For test purpose only!
        n_docs, vocab_size = matrix.shape
        matrix = matrix.toarray().copy()
        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) #C_D_T Count document topic
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size)) #Topic size * word size
        self.nm = np.zeros(n_docs) # Number of documents
        self.nz = np.zeros(self.n_topics) #Number of topic
        self.topics = {} #Topics dictionary

        for m in range(n_docs):
            print(m)
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(self.word_indices(matrix[m, :])):
                
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics) #Randomise the topics
                self.nmz[m,z] += 1 #Distribute the count of topic doucment
                self.nm[m] += 1 #Count the number of occurrences
                self.nzw[z,w] += 1 #Counts the number of topic word distribution
                self.nz[z] += 1 #Distribute the counts of the number of topics
                self.topics[(m,i)] = z #Memorise the correspondence between topics and the entities

    def _conditional_distribution(self, m, w): #Maybe the computation valuables
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size) #Corresponding to the left hand side of the equation 
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics) #Corresponding to the right hand side of the equation
        #We might need to have the section "Word_Concept: like"
        # p_e_c = some expression
        p_z = left * right #* P(e|c)
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1] #Vocabulary size
        n_docs = self.nmz.shape[0] #Document size
        lik = 0 #The calculation of likelihood

        for z in range(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in range(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        #Not necessary values for the calculation
        #V = self.nzw.shape[1]
        num = self.nzw + self.beta #Calculate the counts of the number, the beta is the adjust ment value for the calcluation
        num /= np.sum(num, axis=1)[:, np.newaxis] #Summation of all value and then, but weight should be calculated in this case....
        return num
    
    def theta(self):
        
#        T = self.nmz.shape[0]
        num = self.nmz + self.alpha #Cal
        num /= np.sum(num, axis=1)[:, np.newaxis] 
        
        return num
    
    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        #Gibbs sampling program
        self.phi_set = [] #Storing all results Initialisation of all different models
        self.theta_set = [] #Storing all document_topic relation results & initalisation of all other models
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)
        matrix = matrix.toarray().copy()
        for it in range(maxiter):
            print(it)
            for m in range(n_docs):
                
                for i, w in enumerate(self.word_indices(matrix[m, :])):
                    
                    z = self.topics[(m,i)] #The entities of topics, the value c needs to be included in here
                    self.nmz[m,z] -= 1#Removing the indices 
                    self.nm[m] -= 1 #Removign the indices
                    self.nzw[z,w] -= 1 #Removing the indices
                    self.nz[z] -= 1 #Removing the indices

                    p_z = self._conditional_distribution(m, w) #Put the categorical probability on it
                    z = self.sample_index(p_z)

                    self.nmz[m,z] += 1 #Randomly adding the indices based on the calculated probabilities
                    self.nm[m] += 1 #Adding the entity based on the percentage
                    self.nzw[z,w] += 1 #Addign the entity for 
                    self.nz[z] += 1 #Count the number of the occureences
                    self.topics[(m,i)] = z #Re=assignm the topic

            # FIXME: burn-in and lag!
            print("iteration: {}".format(it))
            print("phi: {}".format(self.phi()))
            print("Theta: {}".format(self.theta()))
            
            self.phi_set.append(self.phi())
            self.theta_set.append(self.theta())
        
        self.maxiter = maxiter
        
    
    #Testing the programs for displaying the data
    #Testing
    
    def set_the_rankings(self, feature_names, doc_names, categories):
        
        self.word_ranking = []
        self.doc_ranking = []
        
#        if(not (self.phi_set in locals() or self.phi.set in globals())):
#            print("The calculation of phi or theta is not done yet!")
        
        for i in range(self.maxiter):
            
            #Calcualte the topic_word distribution
            temp = np.argsort(self.phi_set[i])
            #Calculate the topic_document distribuiton
            temp2 = np.argsort(self.theta_set[i].T)
            
            #Create the leaderships 
            self.word_ranking = [[[topic, ranking, feature_names[idx]] for ranking, idx in enumerate(word_idx)]
                                    for topic, word_idx in enumerate(temp)]
            
            self.doc_ranking = [[[topic, ranking, doc_names[doc_idx], categories[doc_idx]] for ranking, doc_idx in enumerate(docs_idx)]
                        for topic, docs_idx in enumerate(temp2)]
    
    def show_doc_topic_ranking(self, rank=10):
#        if(not (self.doc_ranking in locals() or self.doc_ranking in globals())):
#            print("The calculation of phi or theta is not done yet!")
        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            for j in range(rank):
                 print("Rank: {}, Document: {}, Category: {}".format(self.doc_ranking[i][j][1], self.doc_ranking[i][j][2],
                                                                     self.doc_ranking[i][j][3]))
        
        
    def show_word_topic_ranking(self, rank=10):
#        if(not (self.word_ranking in locals() or self.word_ranking in globals())):
#            print("The calculation of phi or theta is not done yet!")
        
        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            for j in range(rank):
                print("Rank: {}, Word: {}".format(self.word_ranking[i][j][1], self.word_ranking[i][j][2]))
            
        
    
feature_names = R8_training_tf_feature_names
t = LDA(8)


#t._initialize(R8_training_tf_matrix)
#R8_training_tf_feature_names[-5:-1]

t.run(R8_training_tf_matrix, 10)            
           # np.argsor
t.set_the_rankings(feature_names, training_doc_name, training_cat)

#t.word_ranking

t.show_doc_topic_ranking(20)
t.show_word_topic_ranking()



t.phi_set
t.theta_set        
#    def show_all_the_results(self,feature_names, maxiter=30):
#        
#        for i in:
#            np.argsort()
    
if __name__ == "__main__":
    import os
    import shutil

    N_TOPICS = 10
    DOCUMENT_LENGTH = 100
    FOLDER = "topicimg"

    def vertical_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a vertical bar.
        """
        m = np.zeros((width, width))
        m[:, topic_index] = int(document_length / width)
        return m.flatten()

    def horizontal_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a horizontal bar.
        """
        m = np.zeros((width, width))
        m[topic_index, :] = int(document_length / width)
        return m.flatten()

    def save_document_image(filename, doc, zoom=2):
        """
        Save document as an image.
        doc must be a square matrix
        """
        height, width = doc.shape
        zoom = np.ones((width*zoom, width*zoom))
        # imsave scales pixels between 0 and 255 automatically
        sp.misc.imsave(filename, np.kron(doc, zoom))

    def gen_word_distribution(n_topics, document_length):
        """
        Generate a word distribution for each of the n_topics.
        """
        width = n_topics / 2
        vocab_size = width ** 2
        m = np.zeros((n_topics, vocab_size))

        for k in range(width):
            m[k,:] = vertical_topic(width, k, document_length)

        for k in range(width):
            m[k+width,:] = horizontal_topic(width, k, document_length)

        m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

        return m

    def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH, alpha=0.1):
        """
        Generate a document:
            1) Sample topic proportions from the Dirichlet distribution.
            2) Sample a topic index from the Multinomial with the topic
               proportions from 1).
            3) Sample a word from the Multinomial corresponding to the topic
               index from 2).
            4) Go to 2) if need another word.
        """
        theta = np.random.mtrand.dirichlet([alpha] * n_topics)
        v = np.zeros(vocab_size)
        for n in range(length):
            z = sample_index(theta)
            w = sample_index(word_dist[z,:])
            v[w] += 1
        return v

