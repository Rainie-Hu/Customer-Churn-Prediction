# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:26:07 2019

@author: raini
"""

import pandas as pd

#for split the sentences and data
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

#for stemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
 
#for stop words
from sklearn.feature_extraction.text import CountVectorizer

#for feature select
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

#for calculate TFIDF
from sklearn.feature_extraction.text import TfidfTransformer

#for construct models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


customers_file = pd.read_csv('Customers.csv')
customers_data = pd.DataFrame(customers_file)

comments_file = pd.read_csv('Comments.csv')
comments_data = pd.DataFrame(comments_file)
print(customers_data.shape)
print(comments_data.shape)


#------------------------------------------------------------------------------
#split the sentences to lists of words
comments_data['CommentsTokenized'] = comments_data['Comments'].apply(word_tokenize)
#print(comments_data.shape)

#Use SnowballStemmer-------------------------------
stemmer1 = SnowballStemmer('english')
comments_data_stem1 = pd.DataFrame(comments_data['ID'])
comments_data_stem1['CommentsStemmed'] = comments_data['CommentsTokenized'].apply(lambda x: [stemmer1.stem(y) for y in x])
#print(comments_data_stem1.head())

comments_data_stem1['CommentsStemmed'] = comments_data_stem1['CommentsStemmed'].apply(lambda x: " ".join(x))

#Do Bag-Of-Words model - Term - Document Matrix
count_vect = CountVectorizer()
TD_counts1 = count_vect.fit_transform(comments_data_stem1.CommentsStemmed)
print(TD_counts1.shape)


#Use PorterStemmer---------------------------------
stemmer2 = PorterStemmer()
comments_data_stem2 = pd.DataFrame(comments_data['ID'])
comments_data_stem2['CommentsStemmed'] = comments_data['CommentsTokenized'].apply(lambda x: [stemmer2.stem(y) for y in x])
comments_data_stem2['CommentsStemmed'] = comments_data_stem2['CommentsStemmed'].apply(lambda x: " ".join(x))

#Do Bag-Of-Words model - Term - Document Matrix

TD_counts2 = count_vect.fit_transform(comments_data_stem2.CommentsStemmed)
print(TD_counts2.shape)

#Use LancasterStemmer---------------------------------
stemmer3 = LancasterStemmer()
comments_data_stem3 = pd.DataFrame(comments_data['ID'])
comments_data_stem3['CommentsStemmed'] = comments_data['CommentsTokenized'].apply(lambda x: [stemmer3.stem(y) for y in x])
comments_data_stem3['CommentsStemmed'] = comments_data_stem3['CommentsStemmed'].apply(lambda x: " ".join(x))

#Do Bag-Of-Words model - Term - Document Matrix

TD_counts3 = count_vect.fit_transform(comments_data_stem3.CommentsStemmed)
print(TD_counts3.shape)

print(comments_data_stem3.head())
print(comments_data_stem2.head())
print(comments_data_stem1.head())

#Eliminate stop words------------------------------
count_vect_stop = CountVectorizer(stop_words='english', lowercase=False)
TD_counts1_stop = count_vect_stop.fit_transform(comments_data_stem1.CommentsStemmed)
print(TD_counts1_stop.shape)
print(count_vect_stop.get_feature_names())
DF_TD_counts1_stop = pd.DataFrame(TD_counts1_stop.toarray())
print(DF_TD_counts1_stop)

export_csv = DF_TD_counts1_stop.to_csv('DF_TD_counts1_stop.csv')

#compute TF-IDF matrix
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(DF_TD_counts1_stop)
tfidf.shape
DF_tfidf = pd.DataFrame(tfidf.toarray())
print(DF_tfidf.head())

export_cvs = DF_tfidf.to_csv('DF_tfidf.csv')

#Merge files
combined = pd.concat([customers_data, DF_tfidf], axis=1)
print(combined.shape)
#Do one Hot encoding for categorical features
category_col = ['Sex','Status','Children','Car_Owner','RatePlan','Dropped','Paymethod','LocalBilltype','LongDistanceBilltype']
combined_encoding = pd.get_dummies(combined,columns=category_col)
print(combined_encoding.shape)

export_cvs = combined_encoding.to_csv('combined_encoding.csv')

#------------------------------------------------------------------------------
#Feature Selection for combined data
#Filter
combined_encoding_T = combined_encoding['TARGET']
combined_encoding_noT = combined_encoding.drop(['TARGET'],axis=1)
new_combined_encoding = pd.DataFrame(SelectKBest(score_func=chi2, k=150).fit_transform(combined_encoding_noT,combined_encoding_T))
print(new_combined_encoding.shape)

#construct Random forest
rf = RandomForestClassifier()
rf_data = rf.fit(combined_encoding_noT,combined_encoding_T)
print("Accuracy Score: {0:.6f}".format(rf.score(combined_encoding_noT,combined_encoding_T)))
rf_prediction = rf.predict(combined_encoding_noT)
print("Confusion Matrix:")
confusion_matrix(combined_encoding_T,rf_prediction)
classification_report(combined_encoding_T,rf_prediction)

#construct decision tree
dt = DecisionTreeClassifier()
dt_data = dt.fit(combined_encoding_noT,combined_encoding_T)
print("Accuracy Score: {0:.6f}".format(dt.score(combined_encoding_noT,combined_encoding_T)))
dt_prediction = dt.predict(combined_encoding_noT)
print("Confusion Matrix:")
confusion_matrix(combined_encoding_T,dt_prediction)
classification_report(combined_encoding_T,dt_prediction)

#------------------------------------------------------------------------------
#split data to 80% Train 20% Test
train, test = train_test_split(combined_encoding, test_size=0.2, random_state=42)
print(train.shape)
print(test.shape)

x_train = train.drop(columns = 'TARGET')
Y_train = train['TARGET']
x_test = test.drop(columns = 'TARGET')
Y_test = test['TARGET']

#feature selection for split data--------------------
#------------------Filter-----------------
filter_data = SelectKBest(score_func=chi2, k=50).fit(x_train,Y_train)
new_x_train = pd.DataFrame(filter_data.transform(x_train))
new_x_test = pd.DataFrame(filter_data.transform(x_test))
print(new_x_train.shape)
print(new_x_test.shape)

#construct DT-default
dt_filter = DecisionTreeClassifier()
dt_filter_data = dt_filter.fit(new_x_train,Y_train)
dt_filter_prediction = dt_filter.predict(new_x_test)
print("Accuracy Score: {0:.6f}".format(dt_filter.score(new_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,dt_filter_prediction)
classification_report(Y_test,dt_filter_prediction)

#construct DT-Tuning
dt_parameters = {'max_depth': range(5,50,5),'max_leaf_nodes': range(10,50,10),'criterion':['gini','entropy']}
dt_grid = GridSearchCV(dt_filter,dt_parameters)
dt_grid.fit(new_x_train, Y_train)
grid_parm=dt_grid.best_params_
print(grid_parm)

dt_filter_tuning = DecisionTreeClassifier(**grid_parm)
dt_filter_tuning_data = dt_filter_tuning.fit(new_x_train,Y_train)
dt_filter_tuning_prediction = dt_filter_tuning.predict(new_x_test)
print("Accuracy Score: {0:.6f}".format(dt_filter_tuning.score(new_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,dt_filter_tuning_prediction)
classification_report(Y_test,dt_filter_tuning_prediction)

#construct RF-default
rf_filter = RandomForestClassifier()
rf_filter_data = rf_filter.fit(new_x_train,Y_train)
rf_filter_prediction = rf_filter.predict(new_x_test)
print("Accuracy Score: {0:.6f}".format(rf_filter.score(new_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,rf_filter_prediction)
classification_report(Y_test,rf_filter_prediction)

#construct RF-Tuning
rf_parameters = {'max_depth': range(10,50,10),'max_leaf_nodes': range(20,70,10),'criterion':['gini','entropy']}
rf_grid = GridSearchCV(rf_filter,rf_parameters)
rf_grid.fit(new_x_train, Y_train)
grid_parm_rf=rf_grid.best_params_
print(grid_parm_rf)

rf_filter_tuning = RandomForestClassifier(**grid_parm_rf)
rf_filter_tuning_data = rf_filter_tuning.fit(new_x_train,Y_train)
rf_filter_tuning_prediction = rf_filter_tuning.predict(new_x_test)
print("Accuracy Score: {0:.6f}".format(rf_filter_tuning.score(new_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,rf_filter_tuning_prediction)
classification_report(Y_test,rf_filter_tuning_prediction)

#construct GB-default
gb_filter = GradientBoostingClassifier()
gb_filter_data = gb_filter.fit(new_x_train,Y_train)
gb_filter_prediction = gb_filter.predict(new_x_test)
print("accuracy Score (training) for Boosting:{0:6f}".format(gb_filter.score(new_x_test,Y_test)))
print("Confusion Matrix for boosting:")
confusion_matrix(Y_test,gb_filter_prediction)
classification_report(Y_test,gb_filter_prediction)

#construct GB-Tuning
gb_parameters={'n_estimators':[5,10,20],'learning_rate':[0.01,.1,.2],'min_samples_leaf' : range(10,100,10),'max_depth': range(1,10,2)}
gb_grid = RandomizedSearchCV(gb_filter,gb_parameters,n_iter=15)
gb_grid.fit(new_x_train, Y_train)
grid_parm_gb=gb_grid.best_params_
print(grid_parm_gb)

gb_filter_tuning= GradientBoostingClassifier(**grid_parm_gb)
gb_filter_tuning.fit(new_x_train,Y_train)
gb_filter_tuning_prediction = gb_filter_tuning.predict(new_x_test)
print("accuracy Score (training) after hypertuning for Boosting:{0:6f}".format(gb_filter_tuning.score(new_x_test,Y_test)))
print("Confusion Matrix after hypertuning for Boosting:")
print(confusion_matrix(Y_test,gb_filter_tuning_prediction))
print("=== Classification Report ===")
classification_report(Y_test,gb_filter_tuning_prediction)

#------------------Wrapper-----------------
rf_wrapper = RandomForestClassifier(n_estimators=100,n_jobs=-1)
sfs1 = sfs(rf_wrapper,
           k_features=35,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5) 
sfs1 = sfs1.fit(new_x_train, Y_train)
new1_x_train = pd.DataFrame(sfs1.transform(x_train))
new1_x_test = pd.DataFrame(sfs1.transform(x_test))

#Build RF model with selected features
rf_wrapper1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_wrapper1.fit(new1_x_train, Y_train)

rf_wrapper1_prediction = rf_wrapper1.predict(new1_x_test)
print("accuracy Score:{0:6f}".format(rf_wrapper1.score(new1_x_test,Y_test)))
print("Confusion Matrix after hypertuning for Boosting:")
print(confusion_matrix(Y_test,rf_wrapper1_prediction))
print("=== Classification Report ===")
classification_report(Y_test,rf_wrapper1_prediction)

#Build RF-Tuning
rf_wrapper2 = RandomForestClassifier()
rf_parameters1 = {'max_depth': range(10,70,10),'max_leaf_nodes': range(10,70,10),'criterion':['gini','entropy']}
rf_grid1 = GridSearchCV(rf_wrapper2,rf_parameters1)
rf_grid1.fit(new1_x_train, Y_train)
grid_parm_rf1=rf_grid1.best_params_
print(grid_parm_rf1)

rf_wrapper_tuning = RandomForestClassifier(**grid_parm_rf1)
rf_wrapper_tuning_data = rf_wrapper_tuning.fit(new1_x_train,Y_train)
rf_wrapper_tuning_prediction = rf_wrapper_tuning.predict(new1_x_test)
print("Accuracy Score: {0:.6f}".format(rf_wrapper_tuning.score(new1_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,rf_wrapper_tuning_prediction)
classification_report(Y_test,rf_wrapper_tuning_prediction)

#Build DT model with selected features
dt_wrapper1 = DecisionTreeClassifier()
dt_wrapper_data = dt_wrapper1.fit(new1_x_train,Y_train)
dt_wrapper_prediction = dt_wrapper1.predict(new1_x_test)
print("Accuracy Score: {0:.6f}".format(dt_wrapper1.score(new1_x_test,Y_test)))
print("Confusion Matrix:")
confusion_matrix(Y_test,dt_wrapper_prediction)
classification_report(Y_test,dt_wrapper_prediction)

#construct GB-Tuning
gb_parameters1={'n_estimators':[5,10,20],'learning_rate':[0.01,.1,.2],'min_samples_leaf' : range(10,100,10),'max_depth': range(1,10,2)}
gb_grid1 = RandomizedSearchCV(gb_filter,gb_parameters1,n_iter=15)
gb_grid1.fit(new1_x_train, Y_train)
grid_parm_gb1=gb_grid1.best_params_
print(grid_parm_gb1)

gb_wrapper_tuning= GradientBoostingClassifier(**grid_parm_gb1)
gb_wrapper_tuning.fit(new1_x_train,Y_train)
gb_wrapper_tuning_prediction = gb_wrapper_tuning.predict(new1_x_test)
print("accuracy Score (training) after hypertuning for Boosting:{0:6f}".format(gb_wrapper_tuning.score(new1_x_test,Y_test)))
print("Confusion Matrix after hypertuning for Boosting:")
print(confusion_matrix(Y_test,gb_wrapper_tuning_prediction))
print("=== Classification Report ===")
classification_report(Y_test,gb_wrapper_tuning_prediction)








