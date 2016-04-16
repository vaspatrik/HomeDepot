import time
start_time = time.time()
import pandas as pd
import re
import sys  

# encoding="IO-8859-1"
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")


def str_join_words(str1, str2):
    s=(" ").join(["q_"+ z for z in str1.split(" ")])  + " " + str2
    return s

df_all = pd.read_csv('df_all3.csv', index_col=0, encoding="ISO-8859-1")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['search_and_prod_info'] = 'q_'+df_train.search_term+ ' ' + df_train.product_title
print("--- Feature Created: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all.to_csv('df_all4.csv')
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))