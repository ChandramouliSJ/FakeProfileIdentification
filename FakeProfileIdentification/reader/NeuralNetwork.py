
# coding: utf-8

### detect the fake profiles in online social networks using Neural Network

# In[1]:
from django.conf import settings
import sys
import csv
import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gender_guesser.detector as gender
from sklearn.impute import SimpleImputer 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from  sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection  import StratifiedKFold, train_test_split,GridSearchCV,learning_curve    
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc ,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.utilities import percentError


from pybrain.datasets import ClassificationDataSet

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline') 


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


class NeuralNetwork:

####### function for reading dataset from csv files

# In[2]:
  @staticmethod
  def read_datasets(rfile):
    users= pd.read_csv(rfile)    
    x=users
    fake_users=x.query("profile_identification==0")
    genuine_users=x.query("profile_identification==1")
    y=len(fake_users)*[0] + len(genuine_users)*[1]
    return x,y
    


####### function for predicting sex using name of person

# In[3]:
  @staticmethod
  def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'andy':0,'unknown':0,'mostly_male':1, 'male': 2} 
    
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code


####### function for feature engineering

# In[4]:
  @staticmethod
  def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=NeuralNetwork.predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x


####### function for plotting confusion matrix

# In[5]:
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

# In[6]:
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
    plt.savefig(settings.STATIC_ROOT+'\\confusion_matrix_nn.png') 
    plt.close()
    return context


####### Function for training data using Neural Network

# In[7]:
  @staticmethod
  def train(X,y):
    """ Trains and predicts dataset with a Neural Network classifier """

    ds = ClassificationDataSet( len(X.columns), 1,nb_classes=2)
    for k in range(len(X)): 
    	ds.addSample(X.iloc[k],np.array(y[k]))
        
     
        
    #tstdata, trndata = ds.splitWithProportion( 0.20 )
    
    tstdata_temp, trndata_temp = ds.splitWithProportion(0.20)

    tstdata = ClassificationDataSet(7, 1, nb_classes=3)
    for n in range(0, tstdata_temp.getLength()):
       tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

    trndata = ClassificationDataSet(7, 1, nb_classes=3)
    for n in range(0, trndata_temp.getLength()):
       trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
    
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( ) 
    input_size=len(X.columns)
    target_size=1
    hidden_size = 5   
    fnn=None
    if  os.path.isfile('fnn.xml'): 
    	fnn = NetworkReader.readFrom('fnn.xml') 
    else:
    	fnn = buildNetwork( trndata.indim, hidden_size , trndata.outdim, outclass=SoftmaxLayer )	
    trainer = BackpropTrainer( fnn, dataset=trndata,momentum=0.05, learningrate=0.1 , verbose=False, weightdecay=0.01)


    trainer.trainUntilConvergence(verbose = False, validationProportion = 0.15, maxEpochs = 100, continueEpochs = 10 )
    NetworkWriter.writeToFile(fnn, 'oliv.xml')
    predictions=trainer.testOnClassData (dataset=tstdata)
    return tstdata['class'],predictions 


  @staticmethod
  def getNeuralNetwork(rfile):
      t = time.localtime()
      current_time = time.strftime("%H:%M:%S", t)
      context={}
      context['starttime']=current_time

# In[8]: 
      x,y=NeuralNetwork.read_datasets(rfile)
      context['describe']=x.describe()


# In[9]:
 
      x=NeuralNetwork.extract_features(x)
      context['columns']=x.columns
      context['describe']=x.describe() 


# In[10]:
 
      y_test,y_pred =NeuralNetwork.train(x,y)
      output_file = open("oliv.xml", "r")
      context['networkxml']=output_file.read()
      output_file.close()

# In[11]:
      context['accuracy']=accuracy_score(y_test, y_pred) 


# In[12]:

      context['pererror']=percentError(y_pred,y_test)

# In[13]:

      cm=confusion_matrix(y_test, y_pred) 
      context['cm']=cm # without normalization
      NeuralNetwork.plot_confusion_matrix(cm)
 
      cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
      context['cmnormal']=cm_normalized #with normalization
      NeuralNetwork.plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[15]:

      context['clrep']=pd.DataFrame(classification_report(y_test, y_pred, target_names=['Fake','Genuine'],output_dict=True)).transpose().to_html() 


# In[16]:

      s=roc_auc_score(y_test, y_pred)
      context["roc_auc_score"]=s


# In[17]:

      context=NeuralNetwork.plot_roc_curve(y_test, y_pred,context)

      t = time.localtime()
      current_time = time.strftime("%H:%M:%S", t)
      context['endtime']=current_time
      return context