# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:37:27 2019

@author: mehdi
"""
import os 
os.chdir('../Statapps')
os.getcwd()


""" ### Requirements ###"""
import seaborn as sns
import pandas as pd 
import numpy as np
from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()
import matplotlib.pyplot as plt
from matplotlib import pyplot
#Bag Of Words
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
#lemmetization
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from wordcloud import WordCloud

""" ### Vizualisations  ###"""
sns.countplot(data['REPRISE_ACTIVITE'])
sns.countplot(data['REPRISE_INTERLOCUTEUR'])

#Get the distribution of the recommandation ratings
x=data['recommandation_SGK'].value_counts()
x=x.sort_index()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("recommandation_SGK Distribution")
plt.ylabel('# of verbatims', fontsize=12)
plt.xlabel('recommandation ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()




data['recommandation_SGK'].mean()

print('notation supérieure à la moyenne:', data.loc[data['recommandation_SGK'] >data['recommandation_SGK'].mean()].shape[0],
      '; notation inférieure à la moyenne:',data.loc[data['recommandation_SGK'] <= data['recommandation_SGK'].mean()].shape[0])

data_with_stat = data.copy(deep=True)
data_with_stat["Number_of_words"] = data["raisons_recommandation"].apply(lambda x: len(str(x).split()) )
data_with_stat["Number_of_chars"] = data["raisons_recommandation"].apply(lambda x: len(str(x)) )

print('Moyenne de nombre de mots par texte : ',int(data_with_stat['Number_of_words'].mean()))


#Distribution du nombre de mots
plt.figure(figsize=(12, 7))
sns.distplot(data_with_stat.Number_of_words.values, bins=50, kde=False)
plt.xlabel('Nombre de mots')
plt.ylabel('Fréquence')
plt.title("Fréquence du nombre de mots dans le text")
plt.grid(True)
plt.show()



text=data["raisons_recommandation"].dropna().to_frame()
text=text.reset_index()
text['raisons_recommandation2']=['' for i in range(text.shape[0])]
for i in range(text.shape[0]):
    text['raisons_recommandation2'][i]= re.sub(r"[$,.?!\-']"," ",text['raisons_recommandation'][i])
    
    
print(text["raisons_recommandation"][0] , "                                                    ", 
      text["raisons_recommandation2"][0])


text['raisons_recommandation2']=text['raisons_recommandation2'].str.lower() #minuscules
text['Text_letter_only']=['' for i in range(text.shape[0])]
for i in range(text.shape[0]):
    text_splitted = text['raisons_recommandation2'][i].split(' ')
    text_cleaned = filter ((lambda x: re.match(r'^[a-zA-Z].*[a-zA-Z]$',x)),text_splitted)
    text['Text_letter_only'][i]=' '.join(text_cleaned)
    
    

lancaster_stemmer = FrenchStemmer()
text['Text_Stemm']=['' for i in range(text.shape[0])]
for i in range(text.shape[0]):
    text_splitted = text['Text_letter_only'][i].split(' ')
    text_stemmed = [lancaster_stemmer.stem(feature) for feature in text_splitted]
    text['Text_Stemm'][i]=' '.join(text_stemmed)
    
#Elimination des doublons créés par la fonction lancaster_stemmer.stem
text['Text_Stemm_Final']=['' for i in range(text.shape[0])]
for i in range(text.shape[0]):
    text_splitted = text['Text_Stemm'][i].split(' ')
    text_stemmed = list(set(text_splitted))
    text['Text_Stemm_Final'][i]=' '.join(text_stemmed)
    
    
    
from nltk.corpus import stopwords
' , '.join(stopwords.words('french'))




l=','.join(stopwords.words('french'))
L=l.split(',')
type(L)
liste=frozenset(L)




# CountVectorizer implémente la tokenisation et le calcul de l'occurence de chaque token
count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, lowercase = True,
                                   preprocessor=None, stop_words=liste, 
                                   max_features=5000)
bag_of_words = count_vectorizer.fit_transform(text['Text_Stemm_Final'])
features = count_vectorizer1.get_feature_names()


columns_to_drop=['&',
 "''",
 ':',
 ';',
 '`',
'‘',
'’' ,
 '(',
 ')']
bag_of_words_array= bag_of_words.toarray()
text_transformed = pd.DataFrame(data=bag_of_words_array, columns=count_vectorizer.get_feature_names())
text_transformed=text_transformed.drop(columns_to_drop, axis=1)


occurences=text_transformed.sum().to_frame()
occurences=occurences.sort_values([0], ascending=False)


texte = " ".join(recommandation for recommandation in text.Text_Stemm_Final)
print ("Il y a {} mots en combinant toutes les recommendations.".format(len(texte)))


# Create stopword list:
stopwords = liste

# Generate a word cloud image
wordcloud = WordCloud(stopwords=liste, background_color="white").generate(texte)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()





