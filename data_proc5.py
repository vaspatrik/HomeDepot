import time
start_time = time.time()
import pandas as pd
import re

import sys  

# encoding="IO-8859-1"
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

def str_common_word(str1, str2):
    str2 = str2.lower().split(" ")
    if str1 in str2:
        cnt=1
    else:
        cnt=0
    return cnt

def str_common_word2(str1, str2):
    str2 = unicode(str2).lower()
    if str2.find(str1)>=0:
        cnt=1
    else:
        cnt=0
    return cnt

df_all = pd.read_csv('df_all2.csv', encoding="ISO-8859-1")
df_all = df_all[['product_uid','search_term','product_title','product_description']]
df_all.reset_index(inplace=True)
df_attr = pd.read_csv('attributes_stemmed.csv').fillna(" ")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

d_prod_query = {}
for i in range(len(df_all)):
    b_ = unicode(df_all['product_uid'][i])
    if b_ not in d_prod_query:
        d_prod_query[b_] = [list(set(unicode(df_all['search_term'][i]).lower().split(" "))), 
                            unicode(df_all['product_title'][i]),
                            unicode(df_all['product_description'][i])]
    else:
        d_prod_query[b_][0] = list(set(d_prod_query[b_][0] + list(set(unicode(df_all['search_term'][i]).lower().split(" ")))))

f = open("dictionary_stemmed.txt", "w")
f.write(unicode(d_prod_query))
f.close()

print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time)/60),2))
#stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d={}
for i in d_prod_query:
    a = d_prod_query[i][0]
    df_gen_attr = df_attr.loc[df_attr['product_uid'] == unicode(i)+".0"]
    for b_ in a:
        if len(b_)>0:
            col_lst = []
            for j in range(len(df_gen_attr)):
                if str_common_word(b_, df_gen_attr['value'].iloc[j])>0:
                    col_lst.append(df_gen_attr['name'].iloc[j])
            if b_ not in d:
                d[b_] = [1,str_common_word(b_, d_prod_query[i][1]),str_common_word2(b_, d_prod_query[i][1]),col_lst[:]]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, d_prod_query[i][1])
                d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
                d[b_][3] =  list(set(d[b_][3] + col_lst))

ds2 = pd.DataFrame.from_dict(d,orient='index')
ds2.columns = ['count','in title 1','in title 2','attribute type']
ds2 = ds2.sort_values(by=['count'], ascending=[False])

f = open("word_review_stemmed.csv", "w")
f.write("word|count|in title 1|in title 2|attribute type\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + unicode(ds2["count"][i]) + "|" + unicode(ds2["in title 1"][i]) + "|" + unicode(ds2["in title 2"][i]) + "|" + unicode(ds2["attribute type"][i]) + "\n")
f.close()
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))