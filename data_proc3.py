import time
start_time = time.time()

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
import re

import sys  

# encoding="IO-8859-1"
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")


df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_temp = pd.concat((df_train, df_test), axis=0, ignore_index=True)
num_train = df_train.shape[0]
df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)

#v1
dtest = pd.read_csv('word_review_v2.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 1'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ft01','ft02','ft03','ft04','ft05','ft06','ft07','ft08','ft09','ft10','ft11','ft12','ft13','ft14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
#v2
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 2'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ftx01','ftx02','ftx03','ftx04','ftx05','ftx06','ftx07','ftx08','ftx09','ftx10','ftx11','ftx12','ftx13','ftx14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
#v3
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = len(dtest[d]['attribute type'].split(","))
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ftz01','ftz02','ftz03','ftz04','ftz05','ftz06','ftz07','ftz08','ftz09','ftz10','ftz11','ftz12','ftz13','ftz14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)

df_all.to_csv('df_all2.csv')
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))