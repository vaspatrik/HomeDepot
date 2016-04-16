import time
start_time = time.time()

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
import re
import sys

reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

df_all = pd.read_csv('df_all2.csv', encoding="ISO-8859-1", index_col=0)
df_temp = df_all[:]

#v1
dtest = pd.read_csv('word_review_stemmed.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in unicode(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = unicode(b[c-1]).lower()
        #d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 1'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sft01','sft02','sft03','sft04','sft05','sft06','sft07','sft08','sft09','sft10','sft11','sft12','sft13','sft14','sft101','sft102','sft103','sft104','sft105','sft106','sft107','sft108','sft109','sft110','sft111','sft112','sft113','sft114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V1 Complete: %s minutes ---" % round(((time.time() - start_time)/60),2))

#v2
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in unicode(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = unicode(b[c-1]).lower()
        #d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 2'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sftx01','sftx02','sftx03','sftx04','sftx05','sftx06','sftx07','sftx08','sftx09','sftx10','sftx11','sftx12','sftx13','sftx14','sftx101','sftx102','sftx103','sftx104','sftx105','sftx106','sftx107','sftx108','sftx109','sftx110','sftx111','sftx112','sftx113','sftx114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V2 Complete: %s minutes ---" % round(((time.time() - start_time)/60),2))

#v3
[[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in unicode(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = unicode(b[c-1]).lower()
        #d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = len(dtest[d]['attribute type'].split(","))
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sftz01','sftz02','sftz03','sftz04','sftz05','sftz06','sftz07','sftz08','sftz09','sftz10','sftz11','sftz12','sftz13','sftz14','sftz101','sftz102','sftz103','sftz104','sftz105','sftz106','sftz107','sftz108','sftz109','sftz110','sftz111','sftz112','sftz113','sftz114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V3 Complete: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all.to_csv('df_all3.csv')
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))