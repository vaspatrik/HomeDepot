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
from spell_checker import correct
import re

import sys  

# encoding="IO-8859-1"
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

start_time = time.time()
def writeTime():
	global start_time
	print "%s minutes ---" % round(((time.time() - start_time)/60),2)

print 'Reading data'
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")#update here
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1") #update here
df_pro_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1") #update here
df_attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")

writeTime()
print 'Feature engineering'
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

df_bullet = df_attr[df_attr.name.str.startswith(('Bullet01','Bullet02','Bullet03', 'Bullet04'), na=False)][["product_uid", "value"]].rename(columns={"value": "bullet"})
df_bullet.bullet = df_bullet.bullet + ' '
df_bullet = df_bullet.groupby('product_uid').sum().reset_index()

num_train = df_train.shape[0]
df_instances = pd.concat((df_train, df_test), axis=0, ignore_index=True)
dfs = [df_instances, df_pro_desc, df_brand, df_bullet]
df_all =  reduce(lambda left,right: pd.merge(left,right,on='product_uid', how='left'), dfs)

tokenizer = RegexpTokenizer(r'\w+')
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def cleanText(s):
	s = unicode(s).lower()
	if isinstance(s, unicode):
		s = correct(s)
		s = s.replace(" x "," xby ")
		s = s.replace("*"," xby ")
		s = s.replace(" by "," xby")
		s = s.replace("x0"," xby 0")
		s = s.replace("x1"," xby 1")
		s = s.replace("x2"," xby 2")
		s = s.replace("x3"," xby 3")
		s = s.replace("x4"," xby 4")
		s = s.replace("x5"," xby 5")
		s = s.replace("x6"," xby 6")
		s = s.replace("x7"," xby 7")
		s = s.replace("x8"," xby 8")
		s = s.replace("x9"," xby 9")
		s = s.replace("0x","0 xby ")
		s = s.replace("1x","1 xby ")
		s = s.replace("2x","2 xby ")
		s = s.replace("3x","3 xby ")
		s = s.replace("4x","4 xby ")
		s = s.replace("5x","5 xby ")
		s = s.replace("6x","6 xby ")
		s = s.replace("7x","7 xby ")
		s = s.replace("8x","8 xby ")
		s = s.replace("9x","9 xby ")
		s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
		s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
		s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)    
		s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)    
		s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)		
		s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)    
		s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)		
		s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)    
		s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)		
		s = s.replace("whirpool","whirlpool")
		s = s.replace("whirlpoolga", "whirlpool")
		s = s.replace("whirlpoolstainless","whirlpool stainless")
		s = s.replace("  "," ")
		s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
		s = s.lower()
		s = s.replace("  "," ")
		s = s.replace(",","") #could be number / segment later
		s = s.replace("$"," ")
		s = s.replace("?"," ")
		s = s.replace("-"," ")
		s = s.replace("//","/")
		s = s.replace("..",".")
		s = s.replace(" / "," ")
		s = s.replace(" \\ "," ")
		s = s.replace("."," . ")
		s = re.sub(r"(^\.|/)", r"", s)
		s = re.sub(r"(\.|/)$", r"", s)
		s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
		s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
		s = s.replace(" x "," xbi ")
		s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
		s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
		s = s.replace("*"," xbi ")
		s = s.replace(" by "," xbi ")
		s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
		s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
		s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
		s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
		s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
		s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
		s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
		s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
		s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
		s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
		s = s.replace("°"," degrees ")
		s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
		s = s.replace(" v "," volts ")
		s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
		s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
		s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
		s = s.replace("  "," ")
		s = s.replace(" . "," ")
		s = s.replace("toliet","toilet")
		s = s.replace("airconditioner","air conditioner")
		s = s.replace("vinal","vinyl")
		s = s.replace("vynal","vinyl")
		s = s.replace("skill","skil")
		s = s.replace("snowbl","snow bl")
		s = s.replace("plexigla","plexi gla")
		s = s.replace("rustoleum","rust-oleum")
		s = s.replace("whirpool","whirlpool")
		s = s.replace("whirlpoolga", "whirlpool ga")
		s = s.replace("whirlpoolstainless","whirlpool stainless")
		s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
		s = tokenizer.tokenize(s)
		s = [word for word in s if word not in stopwords.words('english')]
		return s


tagger = PerceptronTagger()

def arrayTagger(s):
	taggs = [pos for word,pos in tag._pos_tag(s, None, tagger)]
	return taggs

def stemText(row):
	return [stemmer.stem(x) for x in row]


def seg_words(df):
	str2 = (" ").join(df[1]).lower()
	str2 = re.sub("[^a-z0-9./]"," ", str2)
	str2 = [z for z in set(str2.split()) if len(z)>2]
	words = df[0]
	s = []
	for word in words:
		if len(word)>3:
			s1 = []
			s1 += segmentit(word,str2,True)
			if len(s)>1:
				s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
			else:
				s.append(word)
		else:
			s.append(word)
	return  pd.Series([s])

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def getRatioWhole(df):
	return posWordCount(df[0], df[1], df[2], df[3], True)
def getRatioPos(df):
	return posWordCount(df[0], df[1], df[2], df[3], False)

def posWordCount(search_term, all_data,search_pps, all_pps, all):
	allCount, posCount, nounCount, adjCount = 0, 0, 0, 0
	n,m = len(search_term),len(all_data)
	for i in range(0, n):
		for j in range(0,m):
			if(search_term[i] == all_data[j]):
				allCount += 1
				if (search_pps[i] == all_pps[j]):
					posCount += 1
					nounCount += (search_pps[i].startswith('NN'))
					adjCount += (search_pps[i].startswith('JJ'))
				if (not all):
					break
	return (allCount, posCount, nounCount, adjCount)

def countPos(posArr, pos):
	count = 0
	for p in posArr:
		if (p.startswith(pos)):
			count += 1
	return count

writeTime()
print 'Cleaning text'

df_all['product_title'] = df_all['product_title'].fillna('')
df_all['search_term'] = df_all['search_term'].fillna('')
df_all['product_description'] = df_all['product_description'].fillna('') 
df_all['bullet'] = df_all['bullet'].fillna('') 
df_all['brand'] = df_all['brand'].fillna('') 

df_all['product_title'] = df_all['product_title'].map(lambda x:cleanText(x))
df_all['search_term'] = df_all['search_term'].map(lambda x:cleanText(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:cleanText(x))
df_all['brand'] = df_all['brand'].map(lambda x:cleanText(x))
df_all['bullet'] = df_all['bullet'].map(lambda x:cleanText(x))

df_all['search_term'] = df_all[['search_term', 'product_title']].apply(seg_words, axis=1)

writeTime()
print 'Tagging'
df_all['product_title_pps'] = df_all['product_title'].map(lambda x:arrayTagger(x))
df_all['search_term_pps'] = df_all['search_term'].map(lambda x:arrayTagger(x))
df_all['product_description_pps'] = df_all['product_description'].map(lambda x:arrayTagger(x))
df_all['brand_pps'] = df_all['brand'].map(lambda x:arrayTagger(x))
df_all['bullet_pps'] = df_all['bullet'].map(lambda x:arrayTagger(x))

df_all['search_term_noun_len'] = df_all['search_term_pps'].map(lambda x:countPos(x, 'NN'))
df_all['search_term_adj_len'] = df_all['search_term_pps'].map(lambda x:countPos(x, 'JJ'))

writeTime()
print 'Stemming'
stemmer = SnowballStemmer('english')
df_all['product_title'] = df_all['product_title'].map(lambda row:stemText(row))
df_all['search_term'] = df_all['search_term'].map(lambda row:stemText(row))
df_all['product_description'] = df_all['product_description'].map(lambda row:stemText(row))
df_all['brand'] = df_all['brand'].map(lambda row:stemText(row))
df_all['bullet'] = df_all['bullet'].map(lambda row:stemText(row))
df_all['len_search_term'] = df_all['search_term'].map(lambda x:len(x)).astype(np.int64)
df_all['len_title'] = df_all['product_title'].map(lambda x:len(x)).astype(np.int64)
df_all['len_description'] = df_all['product_description'].map(lambda x:len(x)).astype(np.int64)
df_all['len_brand'] = df_all['brand'].map(lambda x:len(x)).astype(np.int64)

df_all['word_in_title'] = df_all[['search_term', 'product_title','search_term_pps', 'product_title_pps']].apply(getRatioPos, axis=1)
df_all['word_in_description'] = df_all[['search_term', 'product_description','search_term_pps', 'product_description_pps']].apply(getRatioPos, axis=1)
df_all['word_in_brand'] = df_all[['search_term', 'brand','search_term_pps', 'brand_pps']].apply(getRatioPos, axis=1)
df_all['word_in_bullet'] = df_all[['search_term', 'bullet','search_term_pps', 'bullet_pps']].apply(getRatioPos, axis=1)

df_all['word_in_title_pos'] = df_all['word_in_title'].map(lambda x:x[1])
df_all['word_in_title_noun'] = df_all['word_in_title'].map(lambda x:x[2])
df_all['word_in_title_adj'] = df_all['word_in_title'].map(lambda x:x[3])
df_all['word_in_title'] = df_all['word_in_title'].map(lambda x:x[0])

df_all['word_in_description_pos'] = df_all['word_in_description'].map(lambda x:x[1])
df_all['word_in_description_noun'] = df_all['word_in_description'].map(lambda x:x[2])
df_all['word_in_description_adj'] = df_all['word_in_description'].map(lambda x:x[3])
df_all['word_in_description'] = df_all['word_in_description'].map(lambda x:x[0])

df_all['word_in_bullet_pos'] = df_all['word_in_bullet'].map(lambda x:x[1])
df_all['word_in_bullet_noun'] = df_all['word_in_bullet'].map(lambda x:x[2])
df_all['word_in_bullet_adj'] = df_all['word_in_bullet'].map(lambda x:x[3])
df_all['word_in_bullet'] = df_all['word_in_bullet'].map(lambda x:x[0])

df_all['word_in_brand_pos'] = df_all['word_in_brand'].map(lambda x:x[1])
df_all['word_in_brand_noun'] = df_all['word_in_brand'].map(lambda x:x[2])
df_all['word_in_brand_adj'] = df_all['word_in_brand'].map(lambda x:x[3])
df_all['word_in_brand'] = df_all['word_in_brand'].map(lambda x:x[0])

df_all['whole_word_in_title'] = df_all[['search_term', 'product_title','search_term_pps', 'product_title_pps']].apply(getRatioWhole, axis=1)
df_all['whole_word_in_description'] = df_all[['search_term', 'product_description','search_term_pps', 'product_description_pps']].apply(getRatioWhole, axis=1)
df_all['whole_word_in_brand'] = df_all[['search_term', 'brand','search_term_pps', 'brand_pps']].apply(getRatioWhole, axis=1)
df_all['whole_word_in_bullet'] = df_all[['search_term', 'bullet','search_term_pps', 'bullet_pps']].apply(getRatioWhole, axis=1)

df_all['whole_word_in_title_pos'] = df_all['whole_word_in_title'].map(lambda x:x[1])
df_all['whole_word_in_title_noun'] = df_all['whole_word_in_title'].map(lambda x:x[2])
df_all['whole_word_in_title_adj'] = df_all['whole_word_in_title'].map(lambda x:x[3])
df_all['whole_word_in_title'] = df_all['whole_word_in_title'].map(lambda x:x[0])

df_all['whole_word_in_description_pos'] = df_all['whole_word_in_description'].map(lambda x:x[1])
df_all['whole_word_in_description_noun'] = df_all['whole_word_in_description'].map(lambda x:x[2])
df_all['whole_word_in_description_adj'] = df_all['whole_word_in_description'].map(lambda x:x[3])
df_all['whole_word_in_description'] = df_all['whole_word_in_description'].map(lambda x:x[0])

df_all['whole_word_in_bullet_pos'] = df_all['whole_word_in_bullet'].map(lambda x:x[1])
df_all['whole_word_in_bullet_noun'] = df_all['whole_word_in_bullet'].map(lambda x:x[2])
df_all['whole_word_in_bullet_adj'] = df_all['whole_word_in_bullet'].map(lambda x:x[3])
df_all['whole_word_in_bullet'] = df_all['whole_word_in_bullet'].map(lambda x:x[0])

df_all['whole_word_in_brand_pos'] = df_all['whole_word_in_brand'].map(lambda x:x[1])
df_all['whole_word_in_brand_noun'] = df_all['whole_word_in_brand'].map(lambda x:x[2])
df_all['whole_word_in_brand_adj'] = df_all['whole_word_in_brand'].map(lambda x:x[3])
df_all['whole_word_in_brand'] = df_all['whole_word_in_brand'].map(lambda x:x[0])

df_all['ratio_title'] = df_all['word_in_title']/df_all['len_search_term']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_search_term']
df_all['ratio_bullet'] = df_all['word_in_bullet']/df_all['len_search_term']
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_search_term']

df_all['ratio_title_noun'] = df_all['word_in_title_noun']/df_all['search_term_noun_len']
df_all['ratio_description_noun'] = df_all['word_in_description_noun']/df_all['search_term_noun_len']
df_all['ratio_bullet_noun'] = df_all['word_in_bullet_noun']/df_all['search_term_noun_len']
df_all['ratio_brand_noun'] = df_all['word_in_brand_noun']/df_all['search_term_noun_len']

df_all['ratio_title_adj'] = df_all['word_in_title_adj']/df_all['search_term_adj_len']
df_all['ratio_description_adj'] = df_all['word_in_description_adj']/df_all['search_term_adj_len']
df_all['ratio_bullet_adj'] = df_all['word_in_bullet_adj']/df_all['search_term_adj_len']
df_all['ratio_brand_adj'] = df_all['word_in_brand_adj']/df_all['search_term_adj_len']

df_all['ratio_title_pos'] = df_all['word_in_title_pos']/df_all['len_search_term']
df_all['ratio_description_pos'] = df_all['word_in_description_pos']/df_all['len_search_term']
df_all['ratio_bullet_pos'] = df_all['word_in_bullet_pos']/df_all['len_search_term']
df_all['ratio_brand_pos'] = df_all['word_in_brand_pos']/df_all['len_search_term']

df_all[['ratio_title', 'ratio_description', 'ratio_bullet', 'ratio_brand', 'ratio_title_adj','ratio_description_adj', 'ratio_title_noun',
	'ratio_description_noun', 'ratio_bullet_noun','ratio_brand_noun','ratio_bullet_adj',
	'ratio_brand_adj']] = df_all[['ratio_title', 'ratio_description', 'ratio_bullet', 'ratio_brand','ratio_title_adj','ratio_description_adj', 'ratio_title_noun',
	'ratio_description_noun', 'ratio_bullet_noun','ratio_brand_noun','ratio_bullet_adj',
	'ratio_brand_adj']].replace({float('Inf'):0})


df_all['product_title'] = df_all['product_title'].map(lambda x:' '.join(x))
df_all['search_term'] = df_all['search_term'].map(lambda x:' '.join(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:' '.join(x))
df_all['bullet'] = df_all['bullet'].map(lambda x:' '.join(x))
df_all['brand'] = df_all['brand'].map(lambda x:' '.join(x))

df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1
for s in df_brand:
	d[s]=i
	i+=1

df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))



df_all.to_csv('df_all.csv')