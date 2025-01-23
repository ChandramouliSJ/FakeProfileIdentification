from django.conf import settings
# coding: utf-8

### Detect fake profiles in online social networks using Random Forest

# In[54]:

import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sklearn.model_selection as cross_validation
import gender_guesser.detector as gender
from sklearn.impute import SimpleImputer  
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline') 
import time

####### function for reading dataset from csv files

# In[55]:

class RandomForest:

  @staticmethod
  def read_datasets(rfile):
    users= pd.read_csv(rfile)    
    x=users
    fake_users=x.query("profile_identification==0")
    genuine_users=x.query("profile_identification==1")
    y=len(fake_users)*[0] + len(genuine_users)*[1]
    return x,y
    


####### function for predicting sex using name of person

# In[56]:

  @staticmethod
  def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'andy':0,'unknown':0,'mostly_male':1, 'male': 2} 
    
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code


####### function for feature engineering

# In[57]:

  @staticmethod
  def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=RandomForest.predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x


####### function for ploting learning curve

# In[60]:

  @staticmethod
  def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


####### function for plotting confusion matrix

# In[61]:

  @staticmethod
  def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['Fake','Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


####### function for plotting ROC curve

# In[62]:

  @staticmethod
  def plot_roc_curve(y_test, y_pred,context):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    context["falseprate"]=false_positive_rate
    context["trueprate"]=true_positive_rate


    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(settings.STATIC_ROOT+'\\confusion_matrix_rfc_roc.png') 
    plt.close()
    return context


####### Function for training data using Random Forest

# In[63]:

  @staticmethod
  def train(X_train,y_train,X_test,context):
    """ Trains and predicts dataset with a Random Forest classifier """
    
    clf=RandomForestClassifier(n_estimators=40,oob_score=True)
    clf.fit(X_train,y_train)
    context["bestclassifier"]=clf
    # Estimate score
    scores = cross_validation.cross_val_score(clf, X_train,y_train, cv=5)
    context["scores"]=scores
    context["estimatedscore"]= str(round(scores.mean(),5))+' (+/- '+str(round(scores.std() / 2,5))+')'
    title = 'Learning Curves (Random Forest)'
    RandomForest.plot_learning_curve(clf, title, X_train, y_train, cv=5)
   
    plt.savefig(settings.STATIC_ROOT+'\\confusion_matrix_rfc_tain.png') 
    plt.close()
    # Predict 
    y_pred = clf.predict(X_test)
    return  y_pred,context


# In[64]:
  @staticmethod
  def getRandomForest(rfile): 
      t = time.localtime()
      current_time = time.strftime("%H:%M:%S", t)
      context={}
      context['starttime']=current_time

# In[8]: 
      x,y=RandomForest.read_datasets(rfile)
      context['describe']=x.describe()


# In[9]:
 
      x=RandomForest.extract_features(x)
      context['columns']=x.columns
      context['describe']=x.describe() 

 
      X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)
 
      y_pred,context = RandomForest.train(X_train,y_train,X_test,context)

      context['accuracy']=accuracy_score(y_test, y_pred)   

      cm=confusion_matrix(y_test, y_pred) 
      context['cm']=cm # without normalization
      RandomForest.plot_confusion_matrix(cm)
 
      cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
      context['cmnormal']=cm_normalized #with normalization
      RandomForest.plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
 
      context['clrep']=pd.DataFrame(classification_report(y_test, y_pred, target_names=['Fake','Genuine'],output_dict=True)).transpose().to_html() 
 
      context=RandomForest.plot_roc_curve(y_test, y_pred,context)
      
      t = time.localtime()
      current_time = time.strftime("%H:%M:%S", t)
      context['endtime']=current_time
      return context

