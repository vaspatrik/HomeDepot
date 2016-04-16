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

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import pipeline

import sys  

# encoding="IO-8859-1"

reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

start_time = time.time()
def writeTime():
	global start_time
	print "%s minutes ---" % round(((time.time() - start_time)/60),2)

print 'Reading data'
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")#[:100]#update here
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")#[:100] #update here
df_pro_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1") #update here
df_attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")#[:10000]

writeTime()
print 'Feature engineering'
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

df_bullet = df_attr[df_attr.name.str.startswith('Bullet', na=False)][["product_uid", "value"]].rename(columns={"value": "bullet"})
df_bullet.bullet = df_bullet.bullet + ' '
df_bullet = df_bullet.groupby('product_uid').sum().reset_index()

num_train = df_train.shape[0]
df_instances = pd.concat((df_train, df_test), axis=0, ignore_index=True)
dfs = [df_instances, df_pro_desc, df_brand, df_bullet]
df_all =  reduce(lambda left,right: pd.merge(left,right,on='product_uid'), dfs)

tokenizer = RegexpTokenizer(r'\w+')
def cleanText(s):
	s = unicode(s).lower()
	s = tokenizer.tokenize(s)
	s = [word for word in s if word not in stopwords.words('english')]
	return s

tagger = PerceptronTagger()
def arrayTagger(s):
	taggs = [pos for word,pos in tag._pos_tag(s, None, tagger)]
	return taggs

class ColumSelector(BaseEstimator, TransformerMixin):
	def __init__(self, fields):
		self.fields = fields
	def fit(self, x, y=None):
		return self
	def transform(self, full_df):
		return full_df[self.fields]

class ColumTextSelector(BaseEstimator, TransformerMixin):
	def __init__(self, key):
		self.key = key
	def fit(self, x, y=None):
		return self
	def transform(self, full_df):
		return full_df[self.key]

writeTime()
print 'Cleaning text'
df_all['product_title'] = df_all['product_title'].map(lambda x:cleanText(x))
df_all['search_term'] = df_all['search_term'].map(lambda x:cleanText(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:cleanText(x))
df_all['bullet'] = df_all['bullet'].map(lambda x:cleanText(x))
df_all['brand'] = df_all['brand'].map(lambda x:cleanText(x))

writeTime()
print 'Tagging'
df_all['product_title_pps'] = df_all['product_title'].map(lambda x:arrayTagger(x))
df_all['search_term_pps'] = df_all['search_term'].map(lambda x:arrayTagger(x))
df_all['product_description_pps'] = df_all['product_description'].map(lambda x:arrayTagger(x))
df_all['brand_pps'] = df_all['brand'].map(lambda x:arrayTagger(x))
df_all['bullet_pps'] = df_all['bullet'].map(lambda x:arrayTagger(x))

writeTime()
print 'Stemming'
stemmer = SnowballStemmer('english')
df_all['product_title'] = df_all['product_title'].map(lambda row:[stemmer.stem(x) for x in row])
df_all['search_term'] = df_all['search_term'].map(lambda row:[stemmer.stem(x) for x in row])
df_all['product_description'] = df_all['product_description'].map(lambda row:[stemmer.stem(x) for x in row])
df_all['brand'] = df_all['brand'].map(lambda row:[stemmer.stem(x) for x in row])
df_all['bullet'] = df_all['bullet'].map(lambda row:[stemmer.stem(x) for x in row])


df_all['product_title'] = df_all['product_title'].map(lambda x:' '.join(x))
df_all['search_term'] = df_all['search_term'].map(lambda x:' '.join(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:' '.join(x))
df_all['bullet'] = df_all['bullet'].map(lambda x:' '.join(x))
df_all['brand'] = df_all['brand'].map(lambda x:' '.join(x))
df_all['product_title_pps'] = df_all['product_title_pps'].map(lambda x:' '.join(x))
df_all['search_term_pps'] = df_all['search_term_pps'].map(lambda x:' '.join(x))
df_all['product_description_pps'] = df_all['product_description_pps'].map(lambda x:' '.join(x))
df_all['bullet_pps'] = df_all['bullet_pps'].map(lambda x:' '.join(x))
df_all['brand_pps'] = df_all['brand_pps'].map(lambda x:' '.join(x))

df_all.to_csv('df_all_1.csv')

df_train = df_all[~np.isnan(df_all.relevance)]
df_test = df_all[np.isnan(df_all.relevance)]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train.drop(['relevance'], axis=1)
X_test = df_test.drop(['relevance'], axis=1)


rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
gram_vectorizer = CountVectorizer(ngram_range=(1, 5))
#tsvd = TruncatedSVD(n_components=10, random_state = 2016)
clf = pipeline.Pipeline([
		('union', FeatureUnion(
				transformer_list = [
					('txt1', pipeline.Pipeline([('s1', ColumTextSelector(key='search_term')), ('tfidf1', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1'))])),
					('txt2', pipeline.Pipeline([('s2', ColumTextSelector(key='product_title')), ('tfidf2', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1'))])),
					('txt3', pipeline.Pipeline([('s3', ColumTextSelector(key='product_description')), ('tfidf3', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1'))])),
					('txt4', pipeline.Pipeline([('s4', ColumTextSelector(key='brand')),('tfidf4', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1'))])),
					('txt5', pipeline.Pipeline([('s5', ColumTextSelector(key='bullet')),('tfidf5', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1'))])),
					('txt6', pipeline.Pipeline([('s6', ColumTextSelector(key='product_title_pps')), ('gramvec1', CountVectorizer(ngram_range=(2, 2)))])),
					('txt7', pipeline.Pipeline([('s7', ColumTextSelector(key='search_term_pps')), ('gramvec2', CountVectorizer(ngram_range=(2, 2)))])),
					('txt8', pipeline.Pipeline([('s8', ColumTextSelector(key='product_description_pps')), ('gramvec3',CountVectorizer(ngram_range=(2, 2)))])),
					('txt9', pipeline.Pipeline([('s9', ColumTextSelector(key='brand_pps')), ('gramvec4', CountVectorizer(ngram_range=(2, 2)))])),
					('txt10', pipeline.Pipeline([('s10', ColumTextSelector(key='bullet_pps')), ('gramvec5', CountVectorizer(ngram_range=(2, 2)))]))

				],
				transformer_weights = {
						'txt1': 0.8,
						'txt2': 0.5,
						'txt3': 0.5,
						'txt4': 0.6,
						'txt5': 0.6,
						'gramvec1': 0.5,
						'gramvec2': 0.2,
						'gramvec3': 0.2,
						'gramvec4': 0.3,
						'gramvec5': 0.3
						},
				n_jobs = -1
				)), 
		('rfr', rfr)])

writeTime()
print 'Fiting data'
clf.fit(X_train, y_train)

writeTime()
print 'Predicting'
y_pred = clf.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
