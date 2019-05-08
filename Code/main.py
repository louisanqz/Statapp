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
import Tfidf as tfidf

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



#data pour benchmark avec tf idf
X=bow_lasso.transform_bench(pd.concat([train,test]))
y=np.hstack((y_train,y_test))

#Regression lineaire
pred.eval_std(pred.LinearRegression(),X,y,"regression lineaire",n_bootstrap=10)

#8 PPV
pred.eval_std(pred.KNeighborsClassifier(n_neighbors=8),X,y,"8 PPV",n_bootstrap=10)

#Lasso
pred.eval_std(pred.linear_model.LassoCV(cv=3),X,y,"Lasso",n_bootstrap=10)



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


""" #### PREDIRE UN NOUVEAU VERBATIM #### """

ls = pred.load_lstm(chemin_model="predit_model.hdf5",units=64,dims=[50,300]) #charger le meilleurs model

#predire la note d'une nouvelle phrase 
pred.predit_lstm("Je ne suis content",50,mot_model,ls)


""" #### ANALYSE DE L'ERREUR #### """

bags = bow_lasso.BOWT()
bags.fit(train)

X_train_bow = bags.transforme(train)
X_test_bow = bags.transforme(test)

#lasso sur bow 
%time y_pred_train_lassobow,y_pred_test_lassobow = pred.predict_model(pred.linear_model.LassoCV(cv=5),X_train_bow,X_test_bow,y_train,y_test)

#knn sur bow 
%time y_pred_train_knnbow,y_pred_test_knnbow = pred.predict_model(pred.KNNreg(8),X_train_bow,X_test_bow,y_train,y_test)

#reg lineaire sur bow 
%time y_pred_train_regbow,y_pred_test_regbow = pred.predict_model(pred.LinearRegression(),X_train_bow,X_test_bow,y_train,y_test)

#prep pour tfidf 
x_train, x_test, _ , _ = tfidf.prep_data_tfidf(train,test)

#reg lineaire sur tfidf
%time y_pred_train_regtfidf,y_pred_test_regtfidf = tfidf.vect(x_train, x_test, y_train, y_test)

#lasso sur tfidf
%time y_pred_train_lassotfidf,y_pred_test_lassotfidf = tfidf.vect_lasso(x_train, x_test, y_train, y_test)

#knn lineaire sur tfidf
%time y_pred_train_knntfidf,y_pred_test_knntfidf = tfidf.vect_Knn(x_train, x_test, y_train, y_test)


#lasso sur embedding

%time y_pred_train_lassoemb,y_pred_test_lassoemb = pred.predict_model(pred.linear_model.LassoCV(cv=5),ag_train,ag_test,y_train,y_test)

#Knn sur embedding
%time y_pred_train_knnemb,y_pred_test_knnemb = pred.predict_model(pred.KNNreg(8),ag_train,ag_test,y_train,y_test)

#Reg lineraire sur embeding
%time y_pred_train_regemb,y_pred_test_regemb = pred.predict_model(pred.LinearRegression(),ag_train,ag_test,y_train,y_test)


#LSTM
y_pred_train_lstm = ls.predict(X_train[:,:50,:])
y_pred_test_lstm = ls.predict(X_test[:,:50,:])


#Stocke dans un excel pour les comparaisons 
#BOW
train["LASSO_predict_bow"]=y_pred_train_lassobow
test["LASSO_predict_bow"]=y_pred_test_lassobow

train["knn_predict_bow"]=y_pred_train_knnbow
test["knn_predict_bow"]=y_pred_test_knnbow

train["reg_predict_bow"]=y_pred_train_regbow
test["reg_predict_bow"]=y_pred_test_regbow

#TFIDF
train["LASSO_predict_tfidf"]=y_pred_train_lassotfidf
test["LASSO_predict_tfidf"]=y_pred_test_lassotfidf

train["knn_predict_tfidf"]=y_pred_train_knntfidf
test["knn_predict_tfidf"]=y_pred_test_knntfidf

train["reg_predict_tfidf"]=y_pred_train_regtfidf
test["reg_predict_tfidf"]=y_pred_test_regtfidf

#EMB
train["LASSO_predict_emb"]=y_pred_train_lassoemb
test["LASSO_predict_emb"]=y_pred_test_lassoemb

train["knn_predict_emb"]=y_pred_train_knnemb
test["knn_predict_emb"]=y_pred_test_knnemb

train["reg_predict_emb"]=y_pred_train_regemb
test["reg_predict_emb"]=y_pred_test_regemb

train["lstm_predict_emb"]=y_pred_train_lstm
test["lstm_predict_emb"]=y_pred_test_lstm

train.to_excel("train__pred.xlsx")
test.to_excel("test__pred.xlsx")






