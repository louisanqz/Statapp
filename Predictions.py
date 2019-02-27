# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:37:29 2019

@author: mehdi
"""

import os 
os.chdir('../Statapps')
os.getcwd()



""" #### Requirements #### """

import pandas as pd 
import numpy as np
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from preprocessing_word_vectors import load,pad,to_seq


X_train=load("xtrain")
X_test=load("xtest")
y_train=load("ytrain")
y_test=load("ytest")
converted_train=load("convertedtrain")
converted_test=load("convertedtest")
train=pd.read_excel("Train.xlsx")
test=pd.read_excel("Test.xlsx")


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
    
def acc(y_pred):
    res=pd.DataFrame()
    res["Actual"]=y_test
    res["Predicted"]=y_pred
    res["cat_reelle"]=res["Actual"].apply(to_categorie)
    res["cat_predite"]=res["Predicted"].apply(to_categorie)
    acc=sum(res["cat_reelle"]==res["cat_predite"])/len(res)
    return acc

""" Naive """
def stats():
    res={}
    for note in range(11):
        res[note]=sum(train.recommandation_SGK==note)/len(train)
    return res
distri=stats()

def predit(n):
    return np.random.choice([0,1,2,3,4,5,6,7,8,9,10],n,p=list(distri.values()))

y_pred=predit(len(test))
y_test=np.array(test.recommandation_SGK)

res=pd.DataFrame()
res["Actual"]=y_test
res["Predicted"]=y_pred

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

res["cat_reelle"]=res["Actual"].apply(to_categorie)
res["cat_predite"]=res["Predicted"].apply(to_categorie)
sum(res["cat_reelle"]==res["cat_predite"])/len(res)


""" Aggregated-sentence predictions : K-nn, K-means """
def agregate_moy(array,conv):
    """ Averages the words vectors of sentences  """
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

ag_train=agregate_moy(X_train,converted_train)
ag_test=agregate_moy(X_test,converted_test)


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

#REGRESSION LINEAIRE
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(ag_train, y_train)

reg.score(ag_train, y_train)

y_pred=reg.predict(ag_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


#RF
from sklearn.ensemble import RandomForestRegressor
params = {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 30,}

rf = RandomForestRegressor(**params)
rf.fit(ag_train, y_train)

y_pred=rf.predict(ag_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 



#xgb
from sklearn import ensemble
params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 30,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(ag_train, y_train)

y_pred = clf.predict(ag_test) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))



"""### ################### ###"""
"""### LSTM ###"""
"""### ################### ###"""
def construct_model(units=128,dims=[30,300]):
    model = Sequential()
    model.add(LSTM(units, input_shape=(dims[0], dims[1]),dropout=0.5,recurrent_dropout=0.2,return_sequences=True))
    model.add(LSTM(32,dropout=0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])
    return model


w=50
model=construct_model(units=64,dims=[w,300])
filepath="model4 - 2layer.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history=model.fit(X_train[:,:w,:], y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.2,
          callbacks=callbacks_list)


score, accu = model.evaluate(X_test[:,:w,:], y_test)
print(score,accu)

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



""" Reloading best model """

best = Sequential()
best.add(LSTM(64, input_shape=(w, 300),dropout=0.5,recurrent_dropout=0.1,return_sequences=True))
best.add(LSTM(32,dropout=0.5))
best.add(Dense(1, activation='relu'))
best.load_weights("model4 - 2layer.hdf5")
best.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])





"""### ################### ###"""
"""### Result analysis ###"""
"""### ################### ###"""

y_pred=best.predict(X_test[:,:w,:])
y_pred_train=best.predict(X_train[:,:w,:])


res=pd.DataFrame()
res["Actual"]=y_test
res["Predicted"]=y_pred
res.head()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

res["Predicted"].hist(bins=30,normed=True)
plt.show()


res["Actual"].hist(bins=30,normed=True)
plt.show()

res.plot(x="Actual",y="Predicted",style='o')
plt.show()


res["cat_reelle"]=res["Actual"].apply(to_categorie)
res["cat_predite"]=res["Predicted"].apply(to_categorie)
res.head()

sum(res["cat_reelle"]==res["cat_predite"])/len(res)



rest=pd.DataFrame()
rest["Actual_train"]=y_train
rest["Predicted_train"]=y_pred_train
rest.head()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_train))  
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_train))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))) 


rest.plot(x="Actual_train",y="Predicted_train",style='o')
plt.show()




""" Predictions of new sentences  """

mot_model = KeyedVectors.load_word2vec_format('wiki.fr.vec')
def predit(verbatim,maxlen):
    """ Takes a str as input, return a float value"""
    if type(verbatim)!=str:
        return 10
    s,_,_,_=to_seq([verbatim])
    s=pad(s,maxlen)
    return best.predict(np.array(s)[:,:w,:])[0][0]  


