# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:37:29 2019

@author: mehdi
"""

import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import preprocessing_word_vectors as prepro


""" ### ################### ### """
""" ### Benchmarking Models ### """
""" ### ################### ### """
#categories to compare with last year
def to_categorie(x):
    if x<=3.34:
        return "pas satisfait"
    elif 3.34<x<=6.67:
        return "moyennement satisfait"
    else:
        return "satisfait"
    
def acc(y_pred,y_te):
    res=pd.DataFrame()
    res["Actual"]=y_te
    res["Predicted"]=y_pred
    res["cat_reelle"]=res["Actual"].apply(to_categorie)
    res["cat_predite"]=res["Predicted"].apply(to_categorie)
    acc=sum(res["cat_reelle"]==res["cat_predite"])/len(res)
    return acc

""" Naive """
def stats(tr):
    res={}
    for note in range(11):
        res[note]=sum(tr.recommandation_SGK==note)/len(tr)
    return res

def predit(n,tr):
    return np.random.choice([0,1,2,3,4,5,6,7,8,9,10],n,p=list(stats(tr).values()))


def resultats_naif(train,test):
    y_pred=predit(len(test),train)
    y_test=np.array(test.recommandation_SGK)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print("----------------------------")
    print("Accuracy : ",acc(y_pred,y_test))
    return None



""" Aggregated-sentence predictions : K-nn, Lasso, Reg Lin --> Benchmark """

def evaluate_model(predicteur,X_train,X_test,y_train,y_test,nom_predicteur):
    predicteur.fit(X_train,y_train)
    y_pred = predicteur.predict(X_test)
    print("Résultat pour " + nom_predicteur + " :")
    print("----------------------------")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print("----------------------------")
    print("Accuracy : ",acc(y_pred,y_test))
    return None


def eval_std(predicteur,X,y,nom_predicteur,n_bootstrap=50):
    precision=[]
    for boot in range(n_bootstrap):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        predicteur.fit(X_train,y_train)
        y_pred = predicteur.predict(X_test)
        precision.append(acc(y_pred,y_test))
    plt.hist(precision,bins=20,facecolor='g', alpha=0.75)
    plt.xlabel('Précision')
    plt.ylabel('Fréquence')
    plt.title('Precision de '+nom_predicteur+" sur "+str(n_bootstrap)+" echantillons bootstrap")
    plt.grid(True)
    plt.show()
    print("Ecart type estimé pour "+nom_predicteur+" : ",round(np.std(precision),3))
    return None
    


def agregate_moy(array,conv):
    """  Averages the words vectors of sentences  """
    res=np.zeros((array.shape[0],300))
    for i in range(array.shape[0]):
        long=len(conv[i])
        if long > 0:
            res[i]+=array[i,0,:]/long
            for k in range(1,long):
                res[i]+=array[i,k,:]/long
        else :
            pass
    return res


class KNNreg:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        d=metrics.pairwise.pairwise_distances(X,self.X_,metric="cosine")
        y_nei=np.zeros(n_samples)
        for i in range(n_samples):
            idx=np.hstack([d[i,:].reshape(-1,1),self.y_.reshape(-1,1)])
            idxi=idx[idx[:,0].argsort()]
            y_nei[i]=np.mean(idxi[:self.n_neighbors,-1])
        return y_nei


def choose_k_knn(X_train,y_train,X_test,y_test,converted_train,converted_test):
    ag_train=agregate_moy(X_train,converted_train)
    ag_test=agregate_moy(X_test,converted_test)
    x=[]
    y=[]
    for k in range(1,40):
        knn=KNNreg(k)
        knn.fit(ag_train,y_train)
        y_pred = knn.predict(ag_test)
        x.append(k)
        y.append(metrics.mean_absolute_error(y_test, y_pred))
    plt.plot(x,y)
    plt.show()
    print("---------------------------")
    print("Meilleur k :",x[y.index(min(y))])
    return None



"""### ################### ###"""
"""### LSTM ###"""
"""### ################### ###"""



class LSTM_predicteur():
    
    def __init__(self, units=64,dims=[30,300],word_to_use=50):
        self.units = units
        self.dims = dims
        self.w = word_to_use
        

    def fit(self,X_train,y_train,epoch=60):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(self.dims[0], self.dims[1]),dropout=0.5,recurrent_dropout=0.2,return_sequences=False))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
        filepath="model4 - 2layer.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history=model.fit(X_train[:,:self.w,:], y_train,
                          epochs=epoch,
                          batch_size=32,
                          validation_split=0.2,
                          callbacks=callbacks_list)

        """ Visualize training """
        my_dpi=96
        plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        pyplot.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        pyplot.title('model training loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train','validation'], loc='upper left')
        pyplot.show()

    def predict(self,X_test):
        best = Sequential()
        best.add(LSTM(self.units, input_shape=(self.dims[0], self.dims[1]),dropout=0.5,recurrent_dropout=0.2,return_sequences=False))
        best.add(Dense(1, activation='relu'))
        best.load_weights("model4 - 2layer.hdf5")
        best.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])
        return best.predict(X_test[:,:self.w,:])

""" Reloading best model """





"""### ################### ###"""
"""### Result analysis ###"""
"""### ################### ###"""


"""  Predictions of new sentences  """

def predit_lstm(verbatim,maxlen,w2v,chemin_model="model4 - 2layer.hdf5",nb_words=50,units=64,dims=[50,300]):
    best = Sequential()
    best.add(LSTM(units, input_shape=(dims[0],dims[1]),dropout=0.5,recurrent_dropout=0.2,return_sequences=False))
    best.add(Dense(1, activation='relu'))
    best.load_weights(chemin_model)
    best.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])
    if type(verbatim)!=str:
        return 10
    s,_,_,_,_,_=prepro.to_seq([verbatim],w2v)
    s=prepro.pad(s,maxlen)
    return best.predict(np.array(s)[:,:nb_words,:])[0][0]  


