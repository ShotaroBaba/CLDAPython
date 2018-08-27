# -*- coding: utf-8 -*-
"""
@author: Shotaro Baba
"""

import pandas as pd
import numpy as np
import os

import pickle
import requests
import itertools





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





# initialize constants
lemmatizer = WordNetLemmatizer()

#rand = 11

"""Preparation for the pre-processing"""

#Setting stop words 
def define_sw():
    
    #Use default english stopword     
    return set(stopwords.words('english'))


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

#Generate document word count vector
#This vector shows how many number of each word is in
#in each document.
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
def read_test_files(test_path):
    #Construct pandas dataframe
    for_test_purpose_data = pd.DataFrame([], columns=['File', 'Text'])
    training_path = []
    
    #Retrieving the file paths
    for dirpath, dirs, files in os.walk(test_path):
        training_path.extend(files)
    
    #Remove the files other than xml files
    training_path = [x for x in training_path if x.endswith('xml')]
    
    #For each file, xml turns into the text file
    #by parsing the xml file\
    for path_to_file in training_path:
        path_string = os.path.basename(os.path.normpath(path_to_file))        

        file = test_path + '/' + path_string 
        tree = ET.parse(file)
        
        #Turn the string into text file
        root = tree.getroot()
        result = ''
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        #Removing the white space from the result.
        result = result[:-1]
        
                


        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']), ignore_index=True)
    
    return for_test_purpose_data


#Retrieve the data simultaneously
#K represents the range of ranking you want to 
#Retrieve from the Probase database.
def retrieve_p_e_c(feature_names, K = 20):
#    responses  = {}
#    j = 0
    #Asynchronically fetch p(e|c) data from Microsoft probase
    async def retrieve_word_concept_data(feature_names, K = 20):
            #Setting the max workers
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
                #Wait the response from the tasks until the tasks has finished
                for response in await asyncio.gather(*futures):
                    collection_of_results.append(response.json())
                
                return collection_of_results
    #Retrieve the 
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
    #Set your folder path here.
    test_path = "../../R8-Dataset/Dataset/ForTest"
    test_data = read_test_files(test_path)
    vect = generate_vector()
    vectorised_data, feature_names = vectorize(vect, test_data)        

    p_e_c = retrieve_p_e_c(feature_names)
    l = [list(i.keys()) for i in list(p_e_c.values())]
    concept_names = sorted(list(set(itertools.chain.from_iterable(l))))
    
#    concept_sets[len(concept_sets)-1]
    
    #Put the atom concept if there are no concepts in the words
    
    file_lists = test_data['File']
    #Adding atomic elements
    for i in feature_names:
        #if there are no concepts in the words, then...
        if p_e_c[i] == {}:
            
            #Append the words with no related concpets
            #as the atomic concepts
            concept_names.append(i)
    
    #A number of test occurs for generating the good results
    concept_names = sorted(concept_names)

    
    #Create CLDA object
    t = CLDA(vectorised_data, p_e_c, feature_names, concept_names, file_lists, 4, 12)
    
    #Run the methods of CLDA to calculate the topic-document, topic-concept probabilities.
    t.run()
    
 
    #Show all rankings
    t.set_the_rankings()
    t.show_doc_topic_ranking(10)
    t.show_concept_topic_ranking(10)
    t.show_concept_word_ranking(10)

class CLDA(object):
    
    def word_indices(self, vec):
 
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx      
    
    #Initialise the objects in a CLDA instance
    def __init__(self, matrix, concept_dict, feature_names, concept_names, file_lists, n_topics = 3, maxiter = 15, alpha = 0.1, beta = 0.1):

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
        
        #File list
        self.file_lists = file_lists
        
        
        #Beta value
        self.beta = beta
        
        #Max gibb's sampling iteration value
        self.maxiter = maxiter
        
        #Relationship between word and concept
        self.word_concept = {}
        
    
    def _initialize(self):  
     
        #Take the matrix shapes from 
        
        
        n_docs, vocab_size = self.matrix.shape
        
        n_concepts = len(self.concept_names)

        #Relationships between concepts and words
        #Can be used for optimising the processes
        #Make it [[]] for consistency for calculation
        self.concept_word_relationship = [[[self.concept_names.index(y),
                                            self.concept_dict[x][y]]
                                            for y in self.concept_dict[x].keys()] if self.concept_dict[x] != {} else [[self.concept_names.index(x),
                                                                   1.0]] for x in self.feature_names]
        
        self.matrix = self.matrix.toarray().copy()
        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) #C_D_T Count document topic
        # number of times topic z and word w co-occur
        self.nzc = np.zeros((self.n_topics, n_concepts)) #Topic size * word size
        self.nm = np.zeros(n_docs) # The number of documents
        self.nz = np.zeros(self.n_topics) #The number of each topic
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
                    c = np.random.randint(n_concepts) #Randomly distribute the concepts per topics 
                    self.nmz[m,z] += 1 #Distribute the count of topic doucment
                    self.nm[m] += 1 #Count the number of occurrences
                    self.nzc[z,c] += 1 #Counts the number of topic concept distribution
                    self.nz[z] += 1 #Distribute the counts of the number of topics
                    self.topics_and_concepts[(m,i)] = (z,c)
                    self.word_concept[(m,i)] = w #Storing the information about the word
                else:
                    z = np.random.randint(self.n_topics) #Randomise the topics
                    c = self.concept_names.index(self.feature_names[w]) #Randomly distribute the concepts per topics 
                    self.nmz[m,z] += 1 #Distribute the count of topic doucment
                    self.nm[m] += 1 #Count the number of occurrences
                    self.nzc[z,c] += 1 #Counts the number of topic concept distribution
                    self.nz[z] += 1 #Distribute the counts of the number of topics
                    self.topics_and_concepts[(m,i)] = (z,c)
                    self.word_concept[(m,i)] = w #Storing the information about the word
                    

    def _sample_index(self, p_z, nzc):

        
        choice = np.random.multinomial(1,p_z).argmax() #Making the choice throught the 2-dimensional probability array
        concept = int(choice/nzc.shape[0]) #Extract concept array
        topic = choice % nzc.shape[0] #Extract topic index 
        
        #Return topic and concet index
        return topic, concept
    

#    np.array(([2,3,4],[7,6,5])).T.reshape((6,))    
    def _conditional_distribution(self, m, w): #Maybe the computation valuables
       
        concept_size = self.nzc.shape[1]
        n_topics = self.nzc.shape[0]
        p_z_stack = np.zeros((self.nzc.shape[1], self.nzc.shape[0]))
        self.p_z_stack = p_z_stack#For testing purpose
        #Count non-zero value in the 
        #Meaning that the words is the atomic concept
            #Calculate only the 
        for concept_num, concept_prob in self.concept_word_relationship[w]:
            left = (self.nzc[:,concept_num] + self.beta) / (self.nz + self.beta * concept_size)  #Corresponding to the left hand side of the equation 
            right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * n_topics)
            p_z_stack[concept_num] = left * right * concept_prob

            #Normalize the values of p_z_stack
            #Reshaping the probability string 
        return (p_z_stack/np.sum(p_z_stack)).reshape((self.nzc.shape[0] * self.nzc.shape[1],))
        #Calculate the atomic topic distribution calculaiton
        #if there are no positive number in the concept array
        #We might need to have the section "Word_Concept: like"

    #z_c shape is something like this: (topic_num, concept_num)
    def run(self):

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
                    self.nmz[m,z] -= 1#Removing the indices from document topic distribution
                    self.nm[m] -= 1 #Removign the indices from total indices in document
                    self.nzc[z,c] -= 1 #Removing the indices from topic concept distribution
                    self.nz[z] -= 1 #Removing the indices from topic distribution
                    
                    #Calculate the probabilities of both topic and concepts
                    p_z = self._conditional_distribution(m, w) #Put the categorical probability on it
                    #If there are topics, then it returns the random topics based on the 
                    #calculated probabilities, otherwise it returns the atomic bomb
                    z,c = self._sample_index(p_z, self.nzc) #Re-distribute concept and topic 
                    
                    #Randomly adding the indices based on the calculated probabilities
                    self.nmz[m,z] += 1
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
        
        #Storing newest phi value and theta value for calculating word, concept and topic ranking
        self.phi_set.append(self.phi())
        self.theta_set.append(self.doc_prob())

#    
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
        
    #Set the word ranking as it takes too much time to
    def set_the_rankings(self):
        
        self.concept_ranking = []
        self.doc_ranking = []
        
#        if(not (self.phi_set in locals() or self.phi.set in globals())):
#            print("The calculation of phi or theta is not done yet!")
        
        #Calcualte the topic_word distribution by sorti
        temp = np.argsort(-(self.phi_set[0]))
        
        #Calculate the topic_document distribuiton
        temp2 = np.argsort(-(self.theta_set[0].T))
        
        #Create the concept ranking
        self.concept_ranking = [[[topic, ranking, self.concept_names[idx], self.phi_set[0][topic][idx], idx] for ranking, idx in enumerate(concept_idx)]
                                for topic, concept_idx in enumerate(temp)]
        
        #Create the document ranking
        self.doc_ranking = [[[topic, ranking, self.file_lists[doc_idx], self.theta_set[0].T[topic][doc_idx]] for ranking, doc_idx in enumerate(docs_idx)]
                    for topic, docs_idx in enumerate(temp2)]
        
        
        
#Code temporarily exists for the further tests        
#        
#    tmp = [[(topic, ranking, t.concept_names[idx], list(set([(feature_names[t.word_concept[k]]) for k,v in t.topics_and_concepts.items() if v == (topic,idx)  ]))) for ranking, idx in enumerate(concept_idx[0:word_limit])]
#            for topic, concept_idx in enumerate(temp)]
#    
#    temp = np.argsort(-(t.phi_set[0]))
#    tt = [[[topic, ranking, t.concept_names[idx], t.phi_set[0][topic][idx], idx] for ranking, idx in enumerate(concept_idx)]
#                                for topic, concept_idx in enumerate(temp)]
#    tt[0][0:20][0][4]
#    [feature_names[j] for j in list(set([ t.word_concept[k] for k,v in t.topics_and_concepts.items() if v[1] == 5086]))]
#    feature_names[517]
#    t.topics_and_concepts.values((0,5086))
#    t.show
#        self.word_ranking = 
    
    def show_concept_word_ranking(self, K = 10):
        
        #Sorting the values by decending order
        temp = np.argsort(-(self.phi_set[0]))
        
        #Setting the word ranking for each topic
        self.word_ranking = [[(topic, ranking, self.concept_names[idx], list(set([(self.feature_names[self.word_concept[k]]) for k,v in self.topics_and_concepts.items() if v == (topic,idx) ]))) for ranking, idx in enumerate(concept_idx[0:K])]
            for topic, concept_idx in enumerate(temp)]
        
        #Print word ranking over the concept topic
        print('\n')
        print("*********************************")
        print("Word ranking: ")
        print("*********************************")
        print('\n')
        print('\n')
        for elements in self.word_ranking:
            print('\n')
            #Each word ranking over topic and concept is printed
            print("Topic {} word prob ranking: ".format(elements[0][0]))
            for element in elements: 
                
                print("Rank: {}||Concept_name: \"{}\"||Word(s): \"{}\"".format( element[1], element[2], element[3]))
            
        
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
            for j in range(rank):
                 print("Rank: {}, Doc_name: {}, Doc_prob value: {}".format(self.doc_ranking[i][j][1], self.doc_ranking[i][j][2], self.doc_ranking[i][j][3]))
        
        
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
            for j in range(rank):
                print("Rank: {}, Concept: {}, Concept_prob value: {}".format(self.concept_ranking[i][j][1], self.concept_ranking[i][j][2], self.concept_ranking[i][j][3]))

#Baseline method
#Done for comparison           
class LDA(object):

    def __init__(self, n_topics,alpha=0.1, beta=0.1):
       
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
    
    def sample_index(self,p):
        
        return np.random.multinomial(1,p).argmax()
    
    def word_indices(self, vec):
       
        for idx in vec.nonzero()[0]:
            for i in range(int(vec[idx])):
                yield idx
    
    def _initialize(self, matrix):
        
        #For test purpose only!
        n_docs, vocab_size = matrix.shape
#        matrix = matrix.toarray().copy()
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
        

        num = self.nmz + self.alpha 
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

            #Retrieve the phi, theta and document topic co-occurrence probability
            #values
            print("iteration: {}".format(it))
            print("phi: {}".format(self.phi()))
            print("Theta: {}".format(self.theta()))
            print("doc_prob: {}".format(self.doc_prob()))
            
            self.phi_set.append(self.phi())
            self.theta_set.append(self.doc_prob())
        
        self.maxiter = maxiter
        
    
    #Testing the programs for displaying the data
    #Testing
    
    def set_the_rankings(self, feature_names):
        
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
            
            self.doc_ranking = [[[topic, ranking, doc_idx] for ranking, doc_idx in enumerate(docs_idx)]
                        for topic, docs_idx in enumerate(temp2)]
    
    #Show the document ranking over topic
    def show_doc_topic_ranking(self, rank=10):

        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            for j in range(rank):
                 print("Rank: {}, Document: {}".format(self.doc_ranking[i][j][1], self.doc_ranking[i][j][2]))
        
    #Show the topic ranking over the words
    def show_word_topic_ranking(self, rank=10):

        
        for i in range(self.nzw.shape[0]):
            print("#############################")
            print("Topic {} ranking: ".format(i))
            for j in range(rank):
                print("Rank: {}, Word: {}".format(self.word_ranking[i][j][1], self.word_ranking[i][j][2]))
            
        
            

#
if __name__ == "__main__":
#    pass
    main()




