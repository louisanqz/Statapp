# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:08:29 2019

@author: mehdi
"""

import os 
os.chdir('../Statapps')
os.getcwd()

""" #### Requirements #### """
from spellchecker import SpellChecker
spell = SpellChecker(language='fr')
import requests, json


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
        
correct_google("mmodele",typ="all")


""" # Spell checker # """

