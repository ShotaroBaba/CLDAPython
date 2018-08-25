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
import requests
import itertools
import requests
import grequests
from multiprocessing import Pool



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

import asyncio
import concurrent.futures
import requests


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
        for element in root.iter():
            if(element.text != None):
                result += element.text + ' '
        result = result[:-1]
        
                
        #Initialise 
#        for file in generate_files(path):
            #print("Now reading...")
            #print(open(file, "r").read())

        
        for_test_purpose_data = for_test_purpose_data.append(pd.DataFrame([(os.path.basename((path_string)), result)], 
                                                                            columns=['File', 'Text']))
    
    return for_test_purpose_data


#Create the test vectors 






    

#            yield idx


            


#Retrieve all P(e|c) values
def retrieve_p_e_c(feature_names):
    responses  = {}
    j = 0
    if(not os.path.isfile("../../R8-Dataset/Dataset/ForTest/pec_prob_top.pkl")):
        
        #Retrieve the tenth rankings of the words
        K = 10 #Retrieve as much as it could
        for i in feature_names:
            print('Now processing ' + str(j) + " word...")
            j += 1
            #Replace space to + to tolerate the phrases with space
            req_str = 'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' \
            + i.replace(' ', '+') + '&topK=' + str(K)
            response = requests.get(req_str)
            
            #Retrieve the response from json file
            responses[i] = response.json()  
        with open("../../R8-Dataset/Dataset/ForTest/pec_prob_top.pkl", "wb") as f:
            pickle.dump(responses, f)
        return responses
    else:
        with open("../../R8-Dataset/Dataset/ForTest/pec_prob_top.pkl", "rb") as f:
            return pickle.load(f)
        print('File exist... load the feature...')

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
K = 20
test_data = read_test_files()
vect = generate_vector()
vectorise_data, feature_names = vectorize(vect, test_data)     
res_strs = ['https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' + x.replace(' ', '+') + '&topK=' + str(K) for x in feature_names]
res = requests.get(res_strs[66]).content

#res.content
async def time_attack(feature_names, K = 20):
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        collection_of_results = []
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor, 
                requests.get, 
                'https://concept.research.microsoft.com/api/Concept/ScoreByTypi?instance=' + i.replace(' ', '+') + '&topK=' + str(K)
            )
            for i in feature_names
        ]
        for response in await asyncio.gather(*futures):
            collection_of_results.append(response)
        
        return collection_of_results

loop = asyncio.get_event_loop()
future = asyncio.ensure_future(time_attack(feature_names))
results = loop.run_until_complete(future)

temporary = []
for i in results:
    
    temporary.append(i.content)

