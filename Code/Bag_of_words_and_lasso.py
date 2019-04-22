# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:34:06 2019

@author: mehdi
"""


""" #### Requirements #### """

from nltk.stem.snowball import FrenchStemmer
import matplotlib.pyplot as plt
from sklearn import linear_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = FrenchStemmer()
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

import preprocessing_word_vectors as prepro


stop_words_fr = open("stopword_fr.txt", "r")
lines = stop_words_fr.readlines()
lines=[ele.strip('\n') for ele in lines]
stop_words_list=lines
lettres=list("-*.azertyuiopqsdfghjklmwxcvbnAZERTYUIOPQSDFGHJKLMWXCVBNéàèùôïëäêîôÉ!?ç'")

""" #### DATA  #### """
def prep(data):
    data.dropna(inplace=True)
    y=data["recommandation_SGK"].apply(float)
    data["raisons_recommandation"]=data["raisons_recommandation"].apply(str)
    return data,y


""" #### Cleaning stemming  #### """

def clean_sent(sentence):
    p=word_tokenize(sentence)
    aux=list(map(stemmer.stem,p))
    aux2 =list(map(prepro.mot_propre,aux))
    for ele in aux2:
        if type(ele) == list:
            aux2.remove(ele)
            for w in ele:
                aux2.append(w)           
    return " ".join(aux2)
    
    

def transform(data):
    data["clean_sent"]=data["raisons_recommandation"].apply(str).apply(clean_sent)
    count_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, lowercase = True,preprocessor=None, stop_words=stop_words_list, max_features=5000)
    bag_of_words = count_vectorizer.fit_transform(data["clean_sent"])
    features = count_vectorizer.get_feature_names()
    columns_to_drop=['017', '07', '08', '10', '15', '19', '20', '207', '217', '29', '69']
    bag_of_words_array= bag_of_words.toarray()
    text_transformed = pd.DataFrame(data=bag_of_words_array, columns=features)
    text_transformed=text_transformed.drop(columns_to_drop, axis=1)
    text_transformed["sum"]=text_transformed.sum(axis=1)    
    text_transformed["constante"]=1
    return text_transformed,features


def transform_bench(data):
    data["clean_sent"]=data["raisons_recommandation"].apply(str).apply(clean_sent)
    count_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, lowercase = True,preprocessor=None, stop_words=stop_words_list, max_features=5000)
    bag_of_words = count_vectorizer.fit_transform(data["clean_sent"])
    features = count_vectorizer.get_feature_names()
    columns_to_drop=['017', '07', '08', '10', '15', '19', '20', '207', '217', '29', '69']
    bag_of_words_array= bag_of_words.toarray()
    text_transformed = pd.DataFrame(data=bag_of_words_array, columns=features)
    cd=[]
    for c in columns_to_drop:
        if c in text_transformed.columns:
            cd.append(c)
    text_transformed=text_transformed.drop(cd, axis=1)
    text_transformed["sum"]=text_transformed.sum(axis=1)    
    return text_transformed




""" ### LASSO IMPORTANT WORDS ### """

class lasso_viz_imp:
    
    def __init__(self, nb_imp):
        self.nb_imp = nb_imp
        
    def fit(self,data,y):
        self.data_ = data
        self.y_ = y
        return self
    
    def estimate(self):
        alphas, _, coefs = linear_model.lars_path(np.array(self.data_).astype(float),self.y_, method='lasso', verbose=True)
        self.coefs_ = coefs
        self.alphas_ = alphas
        return self 
    
    def viz(self):
        xx = np.sum(np.abs(self.coefs_.T), axis=1)
        xx /= xx[-1]
        plt.figure(figsize=(15,10))
        plt.plot(xx, self.coefs_.T)
        ymin, ymax = plt.ylim()
        plt.vlines(xx, ymin, ymax, linestyle='dashed',alpha=0.2)
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.title('LASSO Path')
        plt.axis('tight')
        plt.show()
        
    def importants(self):
        imp=[]
        for i in range(self.coefs_.shape[0]):
            for k in range(self.nb_imp):
                if self.coefs_ [i,k] != 0:
                    imp.append((self.data_.columns[i],self.coefs_[i,k]))

        return(sorted(imp, key=lambda x: x[1]))








