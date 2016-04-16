# -*- coding: ISO-8859-1 -*-
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer


import sys  
# encoding="IO-8859-1"
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")


def fmean_squared_error(ground_truth, predictions):
	return mean_squared_error(ground_truth, predictions)**0.5

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

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

class FeatureCreator(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self
	def transform(self, data):
		colls=['search_term_noun_len','search_term_adj_len','len_search_term','len_title','len_description','len_brand',
		'word_in_title','word_in_description','word_in_brand',
		'word_in_bullet','word_in_title_pos','word_in_title_noun','word_in_title_adj','word_in_description_pos',
		'word_in_description_noun','word_in_description_adj','word_in_bullet_pos','word_in_bullet_noun',
		'word_in_bullet_adj','word_in_brand_pos','word_in_brand_noun','word_in_brand_adj','whole_word_in_title',
		'whole_word_in_description','whole_word_in_brand','whole_word_in_bullet','whole_word_in_title_pos',
		'whole_word_in_title_noun','whole_word_in_title_adj','whole_word_in_description_pos','whole_word_in_description_noun',
		'whole_word_in_description_adj','whole_word_in_bullet_pos','whole_word_in_bullet_noun','whole_word_in_bullet_adj',
		'whole_word_in_brand_pos','whole_word_in_brand_noun','whole_word_in_brand_adj','ratio_title','ratio_description',
		'ratio_bullet','ratio_brand','ratio_title_noun','ratio_description_noun','ratio_bullet_noun','ratio_brand_noun',
		'ratio_title_adj','ratio_description_adj','ratio_bullet_adj','ratio_brand_adj','ratio_title_pos','ratio_description_pos',
		'ratio_bullet_pos','ratio_brand_pos']
		return data[colls]

start_time = time.time()
def writeTime():
	global start_time
	print "%s minutes ---" % round(((time.time() - start_time)/60),2)

print 'Reading data'
df_all = pd.read_csv('df_all_test.csv', encoding="ISO-8859-1")#[:100]#update here
df_all = df_all.reindex(np.random.permutation(df_all.index))

df_train = df_all[~np.isnan(df_all.relevance)]
df_test = df_all[np.isnan(df_all.relevance)]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train.drop(['relevance'], axis=1)
X_test = df_test.drop(['relevance'], axis=1)

X_train = X_train.dropna()

writeTime()
print 'Building model'
rfr = RandomForestRegressor(n_jobs = 2, random_state = 2016, verbose = 1)
# gbm = GradientBoostingRegressor(random_state = 2016, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
gram_vectorizer = CountVectorizer(ngram_range=(1, 5))
#tsvd = TruncatedSVD(n_components=10, random_state = 2016)
clf = pipeline.Pipeline([
		('union', FeatureUnion(
				transformer_list = [
					('numerical_features',  FeatureCreator()),  
					('txt1', pipeline.Pipeline([('s1', ColumTextSelector(key='search_term')), ('tfidf1', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')), ('tsvd1', TruncatedSVD(n_components=20, random_state = 2016))])),
					('txt2', pipeline.Pipeline([('s2', ColumTextSelector(key='product_title')), ('tfidf2', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')), ('tsvd2', TruncatedSVD(n_components=20, random_state = 2016))])),
					('txt3', pipeline.Pipeline([('s3', ColumTextSelector(key='product_description')), ('tfidf3', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')), ('tsvd3', TruncatedSVD(n_components=20, random_state = 2016))])),
					('txt4', pipeline.Pipeline([('s4', ColumTextSelector(key='brand')),('tfidf4', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')), ('tsvd4', TruncatedSVD(n_components=20, random_state = 2016))])),
					('txt5', pipeline.Pipeline([('s5', ColumTextSelector(key='bullet')),('tfidf5', TfidfVectorizer(ngram_range=(1, 1), stop_words='english', encoding='ISO-8859-1')), ('tsvd5', TruncatedSVD(n_components=20, random_state = 2016))]))
				],
				transformer_weights = {
						'numerical_features': 1,
						'txt1': 0.5,
						'txt2': 0.25,
						'txt3': 0.1,
						'txt4': 0.1,
						'txt4': 0.5
						},
				n_jobs = 2
				)), 
		('rfr', rfr)])


# writeTime()
# N_train = np.arange(3000,10000,1000)
# result_test = np.zeros(len(N_train))
# result_train = np.zeros(len(N_train))
N_test = 50000
N_train = 20000

X_test = X_train[-N_test:]
y_test = y_train[-N_test:]
i = 0;

#'union__transformer_weights' : [[1, 1], [.1, .9], [10, 1000]]
# Best parameters found by grid search:
# {'rfr__max_features': 12, 'rfr__min_samples_split': 1, 'rfr__max_depth': 1, 'rfr__n_estimators': 1000}
# Best CV score:
# -0.531524467571

param_grid = {'rfr__max_features': [12], 'rfr__max_depth': [1,None], 'rfr__n_estimators': [1000], 'rfr__min_samples_split': [1]}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 2, cv = 2, verbose = 20, scoring=RMSE)
model.fit(X_train[:N_train], y_train[:N_train])

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print fmean_squared_error(y_test, y_pred)
y_pred = model.predict(X_train[:N_train])
print fmean_squared_error(y_train[:N_train],y_pred)


# for nt in N_train:
# 	print 'Fiting data', i 
# 	clf.fit(X_train[:nt], y_train[:nt])
# 	writeTime()
# 	print 'Predicting'
# 	y_pred = clf.predict(X_test)
# 	train_pred = clf.predict(X_train[:nt])
# 	result_test[i] = mean_squared_error(y_test, y_pred)**0.5
# 	result_train[i] = mean_squared_error(train_pred, y_train[:nt])**0.5
# 	i = i + 1;

# import matplotlib.pyplot as plt
# plt.xlabel('Size of train set')
# plt.ylabel('Error')
# plt.plot(N_train, result_test, 'red')
# plt.plot(N_train, result_train, 'blue')
# plt.savefig('figure6.png')
# # pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

# importances = clf.named_steps['rfr'].feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.named_steps['rfr'].estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))