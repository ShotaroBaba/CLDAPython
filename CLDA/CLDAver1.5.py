# -*- coding: utf-8 -*-
"""
@author: Shotaro Baba
"""

import pandas as pd
import numpy as np
import os
import csv
import pickle
import requests
import itertools
import requests

from multiprocessing import Pool


import asyncio
import concurrent.futures


import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import xml.etree.ElementTree as ET
#from pyspark.mllib.fpm import FPGrowth
#
#import pyspark

# importing some libraries

#from pyspark.sql import SQLContext

# stuff we'll need for text processing




# initialize constants
lemmatizer = WordNetLemmatizer()

#Set the random state number

rand = 11

"""Preparation for the pre-processing"""

#Setting stop words 
def define_sw():
    
#    stop_word_path = "../../R8-Dataset/Dataset/R8/stopwords.txt"

#    with open(stop_word_path, "r") as f:
#        stop_words = f.read().splitlines()
    
    return set(stopwords.words('english'))# + stop_words)

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
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=[1,1],
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
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        result = result[:-1]
        
                
        #Initialise 
#        for file in generate_files(path):
            #print("Now reading...")
            #print(open(file, "r").read())

        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']), ignore_index=True)
    
    return for_test_purpose_data


#Create the test vectors 






    

#            yield idx


            


#Retrieve all P(e|c) values
#Make sure that the feature values are sorted!
    
#Retrieve the data simultaneously
def retrieve_p_e_c(feature_names):
#    responses  = {}
#    j = 0
    async def retrieve_word_concept_data(feature_names, K = 20):
            with concurrent.futures.ThreadPoolExecutor(max_workers=220) as executor:
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

    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(retrieve_word_concept_data(feature_names))
    results = loop.run_until_complete(future)    
    #Retrieve the tenth rankings of the words
    p_e_c = {}
    
    for idx, i  in enumerate(feature_names):
#    print(i)
#    print(idx)
        p_e_c[i] = results[int(idx)]
        
    
    return p_e_c

#t.phi_set
#t.theta_set        
#    def show_all_the_results(self,feature_names, maxiter=30):
#        
#        for i in:
#            np.argsort()
    

#Now still testing my codes in the
#
def retrieve_data(feature_name):
        K = 10
        print('Now processing ' + str(feature_name) + " word...")

        #Replace space to + to tolerate the phrases with space
        req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
        + feature_name.replace(' ', '+') + '&topK=' + str(K)
        response = requests.get(req_str)
        
        #Retrieve the response from json file
        return response.json()
#



def main():
    test_data = read_test_files()
    vect = generate_vector()
    vectorise_data, feature_names = vectorize(vect, test_data)        
#    feature_names[0:10]
#    print(feature_names)
#    K = 20
#    res_strs = ['https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' +x.replace(' ', '+') + '&topK=' + str(K) for x in feature_names]
#    
#    responses = (grequests.get(u) for u in res_strs)
#    
#    for i in len(responses):
#        if i != None:
#            print(i)
#    p = {}
#    res_strs[-20:-1]
#    reqs = grequests.map(responses)
#    for i, response in enumerate(reqs):
#        
#        if response != None:
#            p[feature_names[i]] = response.content
#        else:
#            p[feature_names[i]] = {}
#    print(p.values())
    
#    [i for i in p_e_c if p_e_c[i] != {}]
#    test_data['File']
    p_e_c = retrieve_p_e_c(feature_names)
    l = [list(i.keys()) for i in list(p_e_c.values())]
    concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
    
#    concept_sets[len(concept_sets)-1]
    
    #Put the atom concept if there are no concepts in the words
    
#    feature_names.index('america')
    #Adding atomic elements
    for i in feature_names:
        #if there are no concepts in the words, then...
        if p_e_c[i] == {}:
            
            #Append the words with no related concpets
            #as the atomic concepts
            concept_names.append(i)
    
    #A number of test occurs for generating the good results
    concept_names = sorted(concept_names)
#    p_e_c_array= np.zeros(shape=(len(feature_names), len(concept_names)))

    #Make it redundant
    #Structure of array
    #p_e_c_array[word][concept]
#    for i in p_e_c.keys():
#        for j in p_e_c[i].keys():
#            p_e_c_array[feature_names.index(i)][concept_names.index(j)] = p_e_c[i][j]
#    file_list =test_data['File']
    #Topic 1
#    file_list[183]
#    file_list[152]
#    file_list[4]
#    file_list[65]
#    file_list[184]
##    
#    #Topic 2
#    file_list[79]
#    file_list[80]
#    file_list[5]
#    file_list[61]
#    file_list[196]
    #For testing purpos
    
    t = CLDA(vectorise_data, p_e_c, feature_names, concept_names, 2, 20)
    
    
    t.run()
#    c = LDA(2)
#    
#    
#    c.run(vectorise_data, 5)
#    t.set_the_rankings()
#    c.set_the_rankings()
#    c.show_concept_topic_ranking(5)
#    c.show_doc_topic_ranking(6)
#    t.show_doc_topic_ranking(10)
#    t.show_concept_topic_ranking(10)

class CLDA(object):
    
    def word_indices(self, vec):
 
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx      
    
    def __init__(self, matrix, concept_dict, feature_names, concept_names, n_topics = 3, maxiter = 15, alpha = 0.1, beta = 0.1):

        #Assign the matrix object
        self.matrix = matrix
        
        #Assign the concept array 
        self.concept_dict = concept_dict
        
        #Assign feature names
        self.feature_names = feature_names
        
        #Assign concept names
        self.concept_names = concept_names
        
        #The number of topics
        self.n_topics = n_topics
        
        #Alpha value
        self.alpha = alpha
        
        #Beta value
        self.beta = beta
        
        #max iteration value
        self.maxiter = maxiter
        
        
        
    
#    def _initialize(self, matrix, concept_array, feature_names, concept_names, n_topics = 3, alpha = 0.1, beta = 0.1):
    def _initialize(self):  
        self.alpha = 0.1
        self.beta = 0.1
        
        #Take the matrix shapes from 
        
        
        n_docs, vocab_size = self.matrix.shape
        
        n_concepts = len(self.concept_names)
        
        self.matrix = self.matrix.toarray().copy()
        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) #C_D_T Count document topic
        # number of times topic z and word w co-occur
        self.nzc = np.zeros((self.n_topics, n_concepts)) #Topic size * word size
        self.nm = np.zeros(n_docs) # Number of documents
        self.nz = np.zeros(self.n_topics) #Number of topic
        self.topics_and_concepts = {} #Topics and concepts
    
        for m in range(n_docs):
            print(m) #Print the document progress
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            #Count the 
            for i, w in enumerate(self. word_indices(self.matrix[m, :])):
                #Randomly put 
                # choose an arbitrary topic as first topic for word i
                # Change this value to if councept_dict[feature_names[w]] != {}
                
                #if the given feature concept probability is zero... then
                if self.concept_dict[self.feature_names[w]] != {}:
                    z = np.random.randint(self.n_topics) #Randomise the topics
                    c = np.random.randint(n_concepts) #Randomly distribute the concepts ver topics 
                    self.nmz[m,z] += 1 #Distribute the count of topic doucment
                    self.nm[m] += 1 #Count the number of occurrences
                    self.nzc[z,c] += 1 #Counts the number of topic concept distribution
                    self.nz[z] += 1 #Distribute the counts of the number of topics
                    self.topics_and_concepts[(m,i)] = (z,c) # P(zd_i, c_d_i) where d is the document location and i is the instance location in the document
                else:
                    z = np.random.randint(self.n_topics) #Randomise the topics
                    c = self.concept_names.index(self.feature_names[w]) #Randomly distribute the concepts ver topics 
                    self.nmz[m,z] += 1 #Distribute the count of topic doucment
                    self.nm[m] += 1 #Count the number of occurrences
                    self.nzc[z,c] += 1 #Counts the number of topic concept distribution
                    self.nz[z] += 1 #Distribute the counts of the number of topics
                    self.topics_and_concepts[(m,i)] = (z,c) # P(zd_i, c_d_i) where d is the document location and i is the instance location in the document
                #Memorise the correspondence between topics and the entities
                                 #Should be put j in somewhere else...
        
#    "aaa".copy()
    def _sample_index(self, p_z, nzc):

        
        choice = np.random.multinomial(1,p_z).argmax() #Making the choice throught the 2-dimensional probability array
        concept = int(choice/nzc.shape[0]) #Extract concept array
        topic = choice % nzc.shape[0] #Extract topic index 
        
        #Return topic and concet index
        return topic, concept
    
#     np.array(([2,3,4],[7,6,5]))/np.sum(np.array(([2,3,4],[7,6,5])))
#    np.reshape(, (6, ))
#    np.array(([2,3,4],[7,6,5])).T.reshape((6,))
#    np.array(([2,3,4],[7,6,5])).T.reshape((6,))    
    def _conditional_distribution(self, m, w): #Maybe the computation valuables
       
        concept_size = self.nzc.shape[1]
        n_topics = self.nzc.shape[0]
        p_z_stack = np.zeros((self.nzc.shape[1], self.nzc.shape[0]))
        self.p_z_stack = p_z_stack#For testing purpose
        #Count non-zero value in the 
        #Meaning that the words is the atomic concept
        if self.concept_dict[self.feature_names[w]] != {}:
            #Calculate only the 
            for c in [self.concept_names.index(c_w) for c_w in self.concept_dict[self.feature_names[w]].keys()]:
                left = (self.nzc[:,c] + self.beta) / (self.nz + self.beta * concept_size)  #Corresponding to the left hand side of the equation 
                right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * n_topics)
#                try:
                    #Make it float just in case
                p_z_stack[c] = left * right * self.concept_dict[self.feature_names[w]][self.concept_names[c]]
#                #If there are no values.. then,
#                except:
#                    p_z_stack[c] = left * right * 0.0
            #Normalize the values of p_z_stack
            #Reshaping the probability string 
            return (p_z_stack/np.sum(p_z_stack)).reshape((self.nzc.shape[0] * self.nzc.shape[1],))
        #Calculate the atomic topic distribution calculaiton
        #if there are no positive number in the concept array
        #Calculate the 
        else:
            #Retrieve the atomic index from the documents 
            atomic_index = self.concept_names.index(self.feature_names[w])
            left = (self.nzc[:,atomic_index] + self.beta) / (self.nz + self.beta * concept_size)
            right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * n_topics)
            p_z_stack[atomic_index] = left * right
            #Normalise p_z value
            
            return (p_z_stack/np.sum(p_z_stack)).reshape((self.nzc.shape[0] * self.nzc.shape[1],))
#        #We might need to have the section "Word_Concept: like"
#        # p_e_c = some expression
#        p_z = left * right #* P(e|c)
#        # normalize to obtain probabilities
#        p_z /= np.sum(p_z)
#        return p_z
    
#    np.array([2,3,4])[[1,2]]
    #np.array(([1,2], [3,4],[5,6])).T.T 
    #z_c shape is something like this: (topic_num, concept_num)
    def run(self, alpha =0.1, beta = 0.1):

        
        self.alpha = alpha
        self.beta = beta
#        self.maxiter = maxiter
        #Gibbs sampling program
        self.phi_set = [] #Storing all results Initialisation of all different models
        self.theta_set = [] #Storing all document_topic relation results & initalisation of all other models
        n_docs, vocab_size = self.matrix.shape
        
        #Initalise the values
        self._initialize()
#        matrix = matrix.toarray().copy()
        for it in range(self.maxiter):
            print(it)
            for m in range(n_docs):
                #Asynchronisation can make the progress faster
                #for conducting gibb sampling algorithm
                for i, w in enumerate(self.word_indices(self.matrix[m, :])):
                    
                    z,c = self.topics_and_concepts[(m,i)] #Extract the topics and concepts from the doucment 
                    self.nmz[m,z] -= 1#Removing the indices 
                    self.nm[m] -= 1 #Removign the indices
                    self.nzc[z,c] -= 1 #Removing the indices
                    self.nz[z] -= 1 #Removing the indices
                    
                    #Calculate the probabilities of both topic and concepts
                    p_z = self._conditional_distribution(m, w) #Put the categorical probability on it
                    #If there are topics, then it returns the random topics based on the 
                    #calculated probabilities, otherwise it returns the atomic bomb
                    z,c = self._sample_index(p_z, self.nzc) #Re-distribute concept and topic 

                    self.nmz[m,z] += 1 #Randomly adding the indices based on the calculated probabilities
                    self.nm[m] += 1 #Adding the entity based on the percentage
                    self.nzc[z,c] += 1 #Addign the entity for 
                    self.nz[z] += 1 #Count the number of the topic occurrences
                    self.topics_and_concepts[(m,i)] = z,c #Re-assign the concepts and topics
            
            #Print the time of the iteration                    
            print("Iteration: {}".format(it))
            #Print phi value(s)
            print("Phi: {}".format(self.phi()))
            #Print the document probability value
            print("Doc_prob: {}".format(self.doc_prob()))
            #Print theta value
            print("Theta: {}".format(self.theta()))
        
        #Storing newest phi value for the calculation
        self.phi_set.append(self.phi())
        self.theta_set.append(self.doc_prob())
        
    #Calculate phi value
    def phi(self):

        #Not necessary values for the calculation
        #V = nzw.shape[1]
        num = self.nzc + self.beta #Calculate the counts of the number, the beta is the adjust ment value for the calcluation
        num /= np.sum(num, axis=1)[:, np.newaxis] #Summation of all value and then, but weight should be calculated in this case....
        return num
    
    #Calculate document probability
    def doc_prob(self):
        
#        T = nmz.shape[0]
        num = self.nmz + self.alpha #Cal
        num /= np.sum(num, axis=1)[:, np.newaxis]
        
        return num
    
    #Calculate theta
    def theta(self):
        num = self.nmz.sum(axis = 0) + self.alpha
        num /= np.sum(num)
        
        return num
        
        
    def set_the_rankings(self):
        
        self.concept_ranking = []
        self.doc_ranking = []
        
#        if(not (self.phi_set in locals() or self.phi.set in globals())):
#            print("The calculation of phi or theta is not done yet!")
        
        #Calcualte the topic_word distribution
        temp = np.argsort(-(self.phi_set[0]))
        #Calculate the topic_document distribuiton
        temp2 = np.argsort(-(self.theta_set[0].T))
        
        #Create the leaderships 
        self.concept_ranking = [[[topic, ranking, self.concept_names[idx], self.phi_set[0][topic][idx]] for ranking, idx in enumerate(concept_idx)]
                                for topic, concept_idx in enumerate(temp)]
        
        self.doc_ranking = [[[topic, ranking, doc_idx, self.theta_set[0].T[topic][doc_idx]] for ranking, doc_idx in enumerate(docs_idx)]
                    for topic, docs_idx in enumerate(temp2)]
    
    def show_doc_topic_ranking(self, rank=10):
#        if(not (self.doc_ranking in locals() or self.doc_ranking in globals())):
#            print("The calculation of phi or theta is not done yet!")
        print('\n')
        print("*********************************")
        print("Theta value: ")
        print("*********************************")
        print(self.theta())
        print('\n')
        
        for i in range(self.nzc.shape[0]):
            print("#############################")
            print("Topic {} doc prob ranking: ".format(i))
            for j in range(rank):
                 print("Rank: {}, Doc_idx: {}, doc_prob value: {}".format(self.doc_ranking[i][j][1], self.doc_ranking[i][j][2], self.doc_ranking[i][j][3]))
        
        
    def show_concept_topic_ranking(self, rank=10):
#        if(not (self.word_ranking in locals() or self.word_ranking in globals())):
#            print("The calculation of phi or theta is not done yet!")
        print('\n')
        print("*********************************")
        print("Phi value: ")
        print("*********************************")
        print(self.phi())
        print('\n')          
        
        for i in range(self.nzc.shape[0]):
            print("#############################")
                    
            print("Topic {} concpet prob ranking: ".format(i))
            for j in range(rank):
                print("Rank: {}, concept: {}, concept_prob value: {}".format(self.concept_ranking[i][j][1], self.concept_ranking[i][j][2], self.concept_ranking[i][j][3]))

#Done for comparison
#Little                 
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

#    def loglikelihood(self):
#        """
#        Compute the likelihood that the model generated the data.
#        """
#        vocab_size = self.nzw.shape[1] #Vocabulary size
#        n_docs = self.nmz.shape[0] #Document size
#        lik = 0 #The calculation of likelihood
#
#        for z in range(self.n_topics):
#            lik += log_multi_beta(self.nzw[z,:]+self.beta)
#            lik -= log_multi_beta(self.beta, vocab_size)
#
#        for m in range(n_docs):
#            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
#            lik -= log_multi_beta(self.alpha, self.n_topics)
#
#        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        #Not necessary values for the calculation
        #V = self.nzw.shape[1]
        num = self.nzw + self.beta #Calculate the counts of the number, the beta is the adjust ment value for the calcluation
        num /= np.sum(num, axis=1)[:, np.newaxis] #Summation of all value and then, but weight should be calculated in this case....
        return num
    
    def doc_prob(self):
        
#        T = self.nmz.shape[0]
        num = self.nmz + self.alpha #Cal
        num /= np.sum(num, axis=1)[:, np.newaxis] 
        
        return num
    
    #Calculate theta
    def theta(self):
        num = self.nmz.sum(axis = 0) + self.alpha
        num /= np.sum(num)
        
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
            print("doc_prob: {}".format(self.doc_prob()))
            
            self.phi_set.append(self.phi())
            self.theta_set.append(self.doc_prob())
        
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
            
        
            

#
if __name__ == "__main__":
    main()




