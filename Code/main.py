# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:35:27 2019

@author: mehdi
"""

""" #### Requirements #### """

import pandas as pd
from gensim.models import KeyedVectors

#Autre fichiers code
import Descriptives_statistics as desc
import preprocessing_word_vectors as prepro
import Bag_of_words_and_lasso as bow_lasso
import Predictions as pred


""" #### DONNEES BRUTES #### """
data=pd.read_excel("verbatims_SRC_230118_ENSAE.xlsx") #données complètes


""" #### STATISTIQUES DESCRIPTIVES #### """
desc.count(data)
desc.distribution_note(data) #distribution des notes
desc.stats_desc(data) #un peu de stats desc
desc.transform_plot(data) #wordcloud avec preprocessing pour enlever les doublons et stemmer les mots

""" #### DONNEES et FICHIERS NECESSAIRES #### """
train=pd.read_excel("Train.xlsx") #un fichier train figé pour pouvoir faire des comparaisons entre méthodes.
test=pd.read_excel("Test.xlsx") #de meme un test figé.
mot_model = KeyedVectors.load_word2vec_format('wiki.fr.vec') #un peu long

#Pour éviter de relancer la mise en vecteur des phrases : 
X_train=prepro.load("xtrain")
X_test=prepro.load("xtest")
y_train=prepro.load("ytrain")
y_test=prepro.load("ytest")
converted_train=prepro.load("converted_train")
converted_test = prepro.load("converted_test")

#Sinon pour recommancer toute la création de la base : 
X_train,X_test,y_train,y_test,converted_train,converted_test = prepro.create_dataset(train,test,mot_model)
#sauvegarder les bases obtenus : 
prepro.dump_all(X_train,X_test,y_train,y_test,converted_train,converted_test)


""" #### BAG OF WORDS ET LASSO POUR VISUALISATION DES VARIABLES IMPORTANTES #### """

data_lasso,y_lasso = bow_lasso.prep(data)
data_lasso_transformed,_ = bow_lasso.transform(data_lasso)
viz_class = bow_lasso.lasso_viz_imp(20) #pour prendre la classe python qui va plot et donner les variables importantes
#selon le chemin de regularisation lasso
viz_class.fit(data_lasso_transformed,y_lasso)
viz_class.estimate()
viz_class.viz() #plot des chemins de regularisation
viz_class.importants() #donne les variables importantes

""" #### PREDICTIONS BENCHMARK ET LSTM #### """

""" BENCHMARKING """
#prediction naive
pred.resultats_naif(train,test)
pred.evaluate_naif(train,test,n_bootstrap=500)
#données agrégé pour algo non lstm : 
ag_train = pred.agregate_moy(X_train,converted_train)
ag_test = pred.agregate_moy(X_test,converted_test)
#KNN
pred.choose_k_knn(X_train,y_train,X_test,y_test,converted_train,converted_test)
pred.evaluate_model(pred.KNNreg(8),ag_train,ag_test,y_train,y_test,"8 plus proches voisins")
pred.eval_std(pred.KNNreg(8),ag_train,y_train,"8 plus proches voisins",n_bootstrap=50)
#Linear regreesion
pred.evaluate_model(pred.LinearRegression(),ag_train,ag_test,y_train,y_test,"Régression linéaire")
pred.eval_std(pred.LinearRegression(),ag_train,y_train,"Régression linéaire",n_bootstrap=50)
#lasso
pred.evaluate_model(pred.linear_model.LassoCV(cv=5),ag_train,ag_test,y_train,y_test,"LASSO")
pred.eval_std(pred.linear_model.LassoCV(cv=5),ag_train,y_train,"LASSO",n_bootstrap=50)

#LSTM
pred.evaluate_model(pred.LSTM_predicteur(units=64,dims=[50,300]),X_train,X_test,y_train,y_test,"LSTM")



""" #### PREDIRE UN NOUVEAU VERBATIM #### """

ls = pred.load_lstm(chemin_model="predi_model.hdf5",units=64,dims=[50,300]) #charger le meilleurs model

#predire la note d'une nouvelle phrase 
pred.predit_lstm("Je ne suis content",50,mot_model,ls)


""" #### ANALYSE DE L'ERREUR #### """

from sklearn import metrics
#Predire sur le train set en entier 
y_pred_train = ls.predict(X_train[:,:50,:])

#Sur le test set
y_pred_test = ls.predict(X_test[:,:50,:])

metrics.mean_squared_error(y_pred_train,y_train)
metrics.mean_absolute_error(y_pred_train,y_train)


metrics.mean_squared_error(y_pred_test,y_test)
metrics.mean_absolute_error(y_pred_test,y_test)







