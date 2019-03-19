# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:34:06 2019

@author: mehdi
"""

""" #### Requirements #### """

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = FrenchStemmer()


""" #### DATA  #### """
data=pd.read_excel("verbatims_SRC_230118_ENSAE.xlsx")
data.dropna(inplace=True)
y=data["recommandation_SGK"].apply(float)
X=data.drop('recommandation_SGK', axis=1)
data["raisons_recommandation"]=data["raisons_recommandation"].apply(str)



""" #### Cleaning stemming  #### """
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

def clean_sent(sentence):
    p=word_tokenize(sentence)
    aux=list(map(stemmer.stem,p))
    aux2 =list(map(mot_propre,aux))
    for ele in aux2:
        if type(ele) == list:
            aux2.remove(ele)
            for w in ele:
                aux2.append(w)           
    return " ".join(aux2)
    
    

data["clean_sent"]=data["raisons_recommandation"].apply(clean_sent)


""" #### Bag of Words  #### """
count_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, lowercase = True,preprocessor=None, stop_words=stop_words_list, max_features=5000)
bag_of_words = count_vectorizer.fit_transform(data["clean_sent"])

features = count_vectorizer.get_feature_names()


columns_to_drop=['017',
 '07',
 '08',
 '10',
 '15',
 '19',
 '20',
 '207',
 '217',
 '29',
 '69']
bag_of_words_array= bag_of_words.toarray()
text_transformed = pd.DataFrame(data=bag_of_words_array, columns=features)
text_transformed=text_transformed.drop(columns_to_drop, axis=1)

text_transformed["sum"]=text_transformed.sum(axis=1)

text_transformed["constante"]=1







""" ### LASSO and IMPORTANT WORDS ### """

""" LASSO predict """
#lasso with CV for alpha choice



""" LASSO importance """

alphas, _, coefs = linear_model.lars_path(np.array(text_transformed).astype(float),y, method='lasso', verbose=True)

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
    for k in range(19):
        if coefs[i,k] != 0:
            imp.append((text_transformed.columns[i],coefs[i,k]))

print(sorted(imp, key=lambda x: x[1]))








