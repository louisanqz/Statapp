# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
""" ### Requirements ### """

import pandas as pd 
import numpy as np
import math
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import preprocessing_word_vectors as prepro
from spellchecker import SpellChecker
import requests, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import Predictions as pred
from scipy import sparse



stop_words_fr = open("stopword_fr.txt", "r")
lines = stop_words_fr.readlines()
lines=[ele.strip('\n') for ele in lines]
stop_words_list=lines
lettres=list("-*.azertyuiopqsdfghjklmwxcvbnAZERTYUIOPQSDFGHJKLMWXCVBNéàèùôïëäêîôÉ!?ç'")

""" # check spell with google # """

def correct_google(text,typ="all"):
    URL="http://suggestqueries.google.com/complete/search?client=firefox&q="+text
    headers = {'User-agent':'Mozilla/5.0'}
    response = requests.get(URL, headers=headers)
    result = json.loads(response.content.decode('utf-8'))
    if typ == "all":
        return result
    if typ == "most likely":
        try:
            return result[1][0]
        except:
            return text

lex = pd.read_csv('http://www.lexique.org/databases/Lexique382/Lexique382.tsv', sep='\t')
lexique=lex["ortho"].tolist()

spell = SpellChecker(language='fr',distance=3)

def orthographe(phrase):
    phrase2=[]
    for word in phrase:
        if word not in lexique:
            corr=correct_google(word,typ="most likely")
            split=corr.split()
            inter=[]
            for i in split:
                if i in list(spell.candidates(word)):
                    inter.append(i)
            if len(inter)==0:
                if len(split)>1:
                    for i in range(len(corr.split())):
                        phrase2.append(corr.split()[i])
                else:
                    phrase2.append(corr)
            else:
                phrase2.append(''.join(inter))
        else:
            phrase2.append(word)
    return phrase2

def list2list(phrase):
    phrase2=[]
    for liste in phrase:
        if type(liste)==list:
            for i in range(len(liste)):
                phrase2.append(liste[i])
        else: 
            phrase2.append(liste)
    return phrase2

def correction(x):
    try :
        x=x.split(' ')
        for i,j in enumerate(x):
            x[i]=prepro.mot_propre(j.lower())
    except:
        if math.isnan(x):
            x=''
    str_list = list(filter(None, list2list(x)))
    return str_list

def to_categorie(x):
    if x<=3.34:
        return "pas satisfait"
    elif 3.34<x<=6.67:
        return "moyennement satisfait"
    else:
        return "satisfait"

#TF IDF SANS STOP WORDS ET 1 GRAM
def vect(x_train, x_test, y_train, y_test):
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.9, sublinear_tf=True, use_idf=True,ngram_range=(1,1),strip_accents='unicode',lowercase=True)
    train_corpus_tf_idf = vectorizer.fit_transform(x_train)
    test_corpus_tf_idf = vectorizer.transform(x_test)
    
    model=LinearRegression()
    model.fit(train_corpus_tf_idf,y_train)
    result = model.predict(test_corpus_tf_idf)
    print("MAE=", mae(result,y_test), "//", "acc=", pred.acc(result,y_test), "MSE=", mse(result,y_test) )


#TF IDF POUR LASSO
    
    
def prep_data_tfidf(train,test):
    train["raisons"]=train["raisons_recommandation"].apply(lambda x: correction(x))
    test["raisons"]=test["raisons_recommandation"].apply(lambda x: correction(x))
    train["clean"]=train["raisons"]
    test["clean"]=test["raisons"]
    train["clean"]=train["clean"].apply(lambda x: " ".join(list2list(x)))
    test["clean"]=test["clean"].apply(lambda x: " ".join(list2list(x)))
    x_train=train['clean']
    y_train=train['recommandation_SGK']
    
    x_test=test['clean']
    y_test=test['recommandation_SGK']
    return x_train, x_test, y_train, y_test
    
def vect_lasso(x_train, x_test, y_train, y_test):
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.9, sublinear_tf=True, use_idf=True,ngram_range=(1,4),strip_accents='unicode',lowercase=True)
    train_corpus_tf_idf = vectorizer.fit_transform(x_train)
    test_corpus_tf_idf = vectorizer.transform(x_test)
    
    train_tf_idf=np.hstack((train_corpus_tf_idf.toarray(), (train_corpus_tf_idf != 0).sum(1) ))
    test_tf_idf=np.hstack((test_corpus_tf_idf.toarray(), (test_corpus_tf_idf != 0).sum(1) ))
    train_tf_idf=sparse.csr_matrix(train_tf_idf)
    test_tf_idf=sparse.csr_matrix(test_tf_idf)

    model=linear_model.Lasso(alpha=0.001)
    model.fit(train_tf_idf,y_train)
    result = model.predict(test_tf_idf)
    print("MAE=", mae(result,y_test), "//", "acc=", pred.acc(result,y_test),"//", "MSE=", mse(result,y_test) )

def viz_lasso(train,test):
    x_train, x_test, y_train, y_test = prep_data_tfidf(train,test)
    X = pd.concat([x_train,x_test],axis=0)
    Y=pd.concat([y_train,y_test],axis=0)
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.9, sublinear_tf=True, use_idf=True,ngram_range=(1,4),strip_accents='unicode',stop_words=stop_words_list,lowercase=True)
    
    corpus_tf_idf = vectorizer.fit_transform(X)
    corpus_tf_idf=np.hstack((corpus_tf_idf.toarray(), (corpus_tf_idf != 0).sum(1) ))
    corpus_tf_idf=sparse.csr_matrix(corpus_tf_idf)
    features = vectorizer.get_feature_names()
    
    features+=["sum"]
    tfidf_array= corpus_tf_idf.toarray()
    text_transformed = pd.DataFrame(data=tfidf_array, columns=features)  
    text_transformed["constante"]=1
    
    alphas, _, coefs = linear_model.lars_path(np.array(text_transformed).astype(float),Y, method='lasso', verbose=True, eps=1e-6)
    
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    plt.figure(figsize=(15,10))
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed',alpha=0.2)
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()
    
    
    imp=[]
    for i in range(coefs.shape[0]):
        for k in range(18):
            if coefs[i,k] != 0:
                imp.append((text_transformed.columns[i],coefs[i,k]))

    imp_u=sorted(imp, key=lambda x: x[1])
    check = set()      
    important = []
    for i in imp_u:
        if i[0] not in check:
            important.append(i)
            check.add(i[0])
    print("Mots importants :")
    return pd.DataFrame(important,columns=["Mot","Coeffcient"])
    
    
    
#TF IDF POUR KNN

def vect_Knn(x_train, x_test, y_train, y_test):
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.9, sublinear_tf=True, use_idf=True,ngram_range=(1,4),strip_accents='unicode',lowercase=True)
    train_corpus_tf_idf = vectorizer.fit_transform(x_train)
    test_corpus_tf_idf = vectorizer.transform(x_test)

    model=KNeighborsRegressor(n_neighbors=2)
    model.fit(train_corpus_tf_idf,y_train)
    result = model.predict(test_corpus_tf_idf)
    print("MAE=", mae(result,y_test), "//", "acc=", pred.acc(result,y_test), "MSE=", mse(result,y_test) )
  






