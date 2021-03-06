# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:37:27 2019

@author: mehdi
"""


""" #### Requirements #### """
import pickle
import numpy as np
import pandas as pd

""" #### DATA and CLEANING #### """
stop_words_fr = open("stopword_fr.txt", "r")
lines = stop_words_fr.readlines()
lines=[ele.strip('\n') for ele in lines]
stop_words_list=lines
lettres=list("-*.azertyuiopqsdfghjklmwxcvbnAZERTYUIOPQSDFGHJKLMWXCVBNéàèùôïëäêîôÉ!?ç'")

def mot_propre(word):
    neg=False
    if "n'" in word:
        neg=True
        word=word.replace("n'","n ")
        li = word.split(' ')[0]
        word=word.split(' ')[1]
    for i in range(len(word)):
        if word[i] not in lettres:
            word=word.replace(word[i],'*')
        if word[i]=='-':
            word=word.replace(word[i],' ')
        if word[i]=="'":
            if i==0:
                word=word.replace(word[i],'*')
            else:
                word=word[0:i-1]+word[i+1:]+"*"+"*"
        if word[i]=='!':
            word=word.replace(word[i],' ! ')
            #i=i-1
        if word[i]=='.':
            word=word.replace(word[i],' . ')
    word=word.replace('*','')
    if neg:
        return [li]+[word]
    else:
        return word

        
""" Word vectors from facebook fast text"""


def to_seq(verbatims,w2v):
    """ Takes a list of str sentences as input
        Returns : 
            a list of word vectors of same length 
            maximum length of vector sentences
            sentences of words actually converted
            number of converted words
            number of words that could not be converted (unknown to w2v dict)"""
    sentence=[]
    sequenc=[]
    maxlen=0
    converted_word=0
    pb_word=0
    for verba in verbatims:
        sen=[]
        aux=verba.split(' ')
        res=[]
        for i in range(len(aux)):
            if '!' in aux[i] or '.' in aux[i] or '?' in aux[i]:
                if type(mot_propre(aux[i]))==list:
                    k=mot_propre(aux[i])[0].split(' ')
                    k=[e for e in k if e!='']
                    res+=k
                    k=mot_propre(aux[i])[1].split(' ')
                    k=[e for e in k if e!='']
                    res+=k
                else:
                    k=mot_propre(aux[i]).split(' ')
                    k=[e for e in k if e!='']
                    res+=k
            else:
                if type(mot_propre(aux[i]))==list:
                    res+=[mot_propre(aux[i])[0]]+[mot_propre(aux[i])[1]]
                else:
                    res+=[mot_propre(aux[i])]
        seq=[]
        for w in res:
            if w not in stop_words_list:
                try:
                    seq.append(w2v[w.lower()])
                    sen.append(w)
                    converted_word+=1
                except:
                    converted_word+=1
                    pb_word+=1
        sequenc.append(seq)
        sentence.append(sen)
        if len(seq)>maxlen:
            maxlen=len(seq)
    return sequenc,maxlen,sentence,[converted_word,pb_word]




def pad(liste,maxlen):
    """ Takes list and int as input
        Returns same list with every element padded to maxlen with zeros-vectors """
    for ele in liste:
        while len(ele) < maxlen :
            ele.append(np.zeros((300,)))
    return liste



#Base_train, Base_test = train_test_split(data[["recommandation_SGK","raisons_recommandation"]],test_size=0.2)

#
    
def create_dataset(tr,te,w2v):
    Base_train, Base_test = tr, te
    X_train,y_train,X_test,y_test=list(Base_train["raisons_recommandation"].apply(str)),list(Base_train["recommandation_SGK"]),list(Base_test["raisons_recommandation"].apply(str)),list(Base_test["recommandation_SGK"])
    X_train,maxlentrain,converted_train,pb_train=to_seq(X_train,w2v)
    X_test,maxlentest,converted_test,pb_test=to_seq(X_test,w2v)
    maxlen=max(maxlentest,maxlentrain)
    #print((pb_train[1]+pb_test[1])/(pb_train[0]+pb_test[0])*100)
    #1.75%
    X_train=pad(X_train,maxlen)
    X_test=pad(X_test,maxlen)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return X_train,X_test,y_train,y_test,converted_train,converted_test

""" pickle saving """


def dump(data,name):
    with open(name, 'wb') as d:
        pickle.dump(data, d)
    return None

def dump_all(X_train,X_test,y_train,y_test,converted_train,converted_test):
    dump(X_train,"xtrain")
    dump(X_test,"xtest")
    dump(y_train,"ytrain")
    dump(y_test,"ytest")
    dump(converted_train,"convertedtrain")
    dump(converted_test,"convertedtest")
    return None


""" pickle loading """


def load(name):
    with open(name, 'rb') as fp:
        file = pickle.load(fp)
    return file
