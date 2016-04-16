# -*- coding: ISO-8859-1 -*-
import time

import numpy as np
import pandas as pd

from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
from nltk import tag

from nltk.stem.snowball import SnowballStemmer 
import re

import sys  

# encoding="IO-8859-1"

reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

start_time = time.time()
def writeTime():
	global start_time
	print "%s minutes ---" % round(((time.time() - start_time)/60),2)

words = ""
def getWords(s):
	global words
	s = unicode(s).lower()
	words += s;

print 'Reading data'
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")#update here
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")[:0] #update here
df_pro_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1") #update here
df_attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")

df_all['product_title'].map(lambda x:getWords(x))
df_all['search_term'].map(lambda x:getWords(x))
df_all['product_description'].map(lambda x:getWords(x))
df_bullet = df_attr[df_attr.name.str.startswith(('Bullet01','Bullet02','Bullet03', 'Bullet04'), na=False)][["product_uid", "value"]].rename(columns={"value": "bullet"})
df_bullet.map(lambda x:getWords(x))
