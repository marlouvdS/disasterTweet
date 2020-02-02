import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from wordcloud import WordCloud
import pandas as pd
from python.dataPrepros import preprocessing, createDict
from sklearn.feature_extraction.text import CountVectorizer
import cudf

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

cv = CountVectorizer(max_features = len(uniqueWords))
#Create Bag of Words Model , here X represent bag of words
X = cv.fit_transform(corpus).todense()
y = ds_train['target'].values

#Split the train data set to train and test data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=2020)
print('Train Data splitted successfully')

# Fitting Gaussian Naive Bayes to the Training set
classifier_gnb = GaussianNB()
classifier_gnb.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_gnb = classifier_gnb.predict(X_test)
# Making the Confusion Matrix
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# Fitting Logistic Regression Model to the Training set
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_lr = classifier_lr.predict(X_test)
# Making the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
#Calculating Model Accuracy
print('Logistic Regression Model Accuracy Score for Train Data set is {}'.format(classifier_lr.score(X_train, y_train)))
print('Logistic Regression Model Accuracy Score for Test Data set is {}'.format(classifier_lr.score(X_test, y_test)))
print('Logistic Regression Model F1 Score is {}'.format(f1_score(y_test, y_pred_lr)))

#Fitting into test set
X_testset=cv.transform(test_df['text']).todense()
#Predict data with classifier created in previous section
y_test_pred_gnb = classifier_gnb.predict(X_testset)
y_test_pred_lr = classifier_lr.predict(X_testset)

#Fetching Id to differnt frame
y_test_id=test_df[['id']]
#Converting Id into array
y_test_id=y_test_id.values
#Converting 2 dimensional y_test_id into single dimension
y_test_id=y_test_id.ravel()

#Converting 2 dimensional y_test_pred for all predicted results into single dimension
y_test_pred_gnb=y_test_pred_gnb.ravel()
y_test_pred_lr=y_test_pred_lr.ravel()

#Creating Submission dataframe
submission_df_gnb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_gnb})
submission_df_lr=pd.DataFrame({"id":y_test_id,"target":y_test_pred_lr})

#Setting index as Id Column
submission_df_gnb.set_index("id")
submission_df_lr.set_index("id")

#Converting into CSV file for submission
submission_df_gnb.to_csv("submission_gnb.csv",index=False)
submission_df_lr.to_csv("submission_lr.csv",index=False)