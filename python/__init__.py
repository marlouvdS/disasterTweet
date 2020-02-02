import matplotlib.pyplot as plt
import nltk
import numpy
import pillow
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from nltk import PorterStemmer, re
from nltk.corpus import stopwords
# from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from python.dataPrepros import preprocessing, createDict

train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

ds_train = train_df.drop(['location'],axis=1)

# creating bool series True for NaN values
bool_series_keyword = pd.isnull(ds_train['keyword'])
#dropping missing 'keyword' records from train data set
ds_train=ds_train.drop(ds_train[bool_series_keyword].index,axis=0)
#Resetting the index after droping the missing records
ds_train=ds_train.reset_index(drop=True)
print("Number of records after removing missing keywords",len(ds_train))

corpus = preprocessing(ds_train)
uniqueWords = createDict(corpus)
uniqueWords=uniqueWords[uniqueWords['WordFrequency']>=20]

wordcloud = WordCloud().generate(" ".join(corpus))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()
print(ds_train.head())