from django.shortcuts import render
from django.conf import settings
import pandas as pd
from django.http import HttpResponse
from django.http import JsonResponse
import matplotlib.pyplot as plt
from pathlib import Path    
import string
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from pybrain.utilities import percentError
from pybrain.datasets import ClassificationDataSet

from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

import gender_guesser.detector as gender
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

plt.switch_backend('agg')

feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code','profile_identification']
afeature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
   

def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'andy':0,'unknown':0,'mostly_male':1, 'male': 2} 
    
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code
    
def extract_features():
    rdf= pd.read_csv(rfilename) 
    x=rdf
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    x=x.loc[:,feature_columns_to_use]
    return x

def aextract_features():
    users= pd.read_csv(rfilename)    
    x=users
    fake_users=x.query("profile_identification==0")
    genuine_users=x.query("profile_identification==1")
    
    y=len(fake_users)*[0] + len(genuine_users)*[1] 
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name']) 
    x=x.loc[:,afeature_columns_to_use]
    return x,y


rfilename="users.csv" 
def reuploadfile(request): 
    context={} 
    df= pd.read_csv("users-sample.csv")  
     
    context = { 
        "head": [], 
    }   
    for v in df.head().to_dict('records'):
       context['head'].append(v.items())   
     
    return render(request, 'reuploadfile.html', context=context)
    
def savefile(request): 
   try:
    file=request.FILES['file']  
    output_file = open(rfilename, "wb")           
    output_file.write(file.read()) 
    output_file.close()
    dt_epoch = datetime.now().timestamp() 
    os.utime(rfilename, (dt_epoch, dt_epoch))    
    
    return JsonResponse({"success":1,"message":"File Uploaded Successfully."});     
   except Exception as e:
    return JsonResponse({"success":0,"message":str(e)});   
 
# Create your views here.
def home(request):
  try: 
    
    context = {
        "success": True,
        "data": { 
            }, 
    } 
    # send the news feed to template in context
    return render(request, 'home.html', context=context)
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})        


def start(request):
  try:
      
    users= pd.read_csv(rfilename)    
    x=users
    fake_users=x.query("profile_identification==0")
    genuine_users=x.query("profile_identification==1")
    
    context = {
        "success": True,
        "data": {
               'genuine_users': len(genuine_users),
               'fake_users':len(fake_users),
               'total':len(genuine_users)+len(fake_users)
            }, 
    } 
    # send the news feed to template in context
    return render(request, 'index.html', context=context)
  except Exception as e:
    return render(request, 'error.html', context={'message':str(e)})        
  
        
  
def explorecount(request):
 
    x=extract_features()
    dcount=x.groupby(['profile_identification'])['profile_identification'].agg(list)
    datacount={'Genuine':len(dcount[1]),'Fake':len(dcount[0])} 
    courses = list(datacount.keys())
    values = list(datacount.values())
    #data.groupby(['label'])['label'].count().plot(kind="bar") 
    fig = plt.figure(figsize = (10, 5))
    plt.bar(courses, values, color ='maroon', width = 0.4) 
    plt.xlabel("Status")
    plt.ylabel("No. of Profiles")  
    plt.savefig(settings.STATIC_ROOT+'\\visualize.png')  
    plt.close() 
    x=x.describe();
    data={}
    for  column in x: 
           for key,val in x[column].items():
             if data.get(key)==None:
                data[key]=[]
             data[key].append(x[column][key])
             
    for key,val in data.items():
       fig = plt.figure().set_figwidth(25)
       plt.bar(feature_columns_to_use, list(val), color ='maroon', width = 0.8) 
       plt.xlabel("Feature")
       plt.ylabel("No. of Profiles")  
       plt.savefig(settings.STATIC_ROOT+'\\visualize-'+key+'.png')  
       plt.close()
    
    context={'datacount':datacount.items(),'feature_columns_to_use':feature_columns_to_use,'data':data.items()}
    
    return render(request, 'explorecount.html', context=context)   
  
 
from .NeuralNetwork import NeuralNetwork
from .RandomForest import RandomForest
from .SupportVectorMachine import SupportVectorMachine
 
def exploreconfusionmatrix(request,atype):  
    if atype=="nn": 
      context=NeuralNetwork.getNeuralNetwork(rfilename)     
    if atype=="rfc": 
      context=RandomForest.getRandomForest(rfilename)     
    if atype=="svm": 
      context=SupportVectorMachine.getSupportVectorMachine(rfilename)     
      
    return render(request, 'exploreconfusionmatrix-'+atype+'.html', context=context)   
    
    
def profilecheck(request): 
    context={} 
    df= pd.read_csv("user-sample.csv")  
     
    context = { 
        "head": [], 
    }   
    for v in df.head().to_dict('records'):
       context['head'].append(v.items())   
     
    return render(request, 'profilecheck.html', context=context)
    
def uploadprofilecheck(request): 
   try:
    file=request.FILES['file1']  
    output_file = open("sinlge-profile-check.csv", "wb")           
    output_file.write(file.read()) 
    output_file.close()  
    users= pd.read_csv(rfilename)    
    x=users
    fake_users=x.query("profile_identification==0")
    genuine_users=x.query("profile_identification==1")

    checkprofile= pd.read_csv("sinlge-profile-check.csv")
    found=""
    if genuine_users.merge(checkprofile,  on='id', how = 'inner' ,indicator=False).shape[0]>0:
        result="Profile is Genuine"
        found="genuine"
    elif fake_users.merge(checkprofile, on='id', how = 'inner' ,indicator=False).shape[0]>0:      
        result="Profile is Fake" 
        found="fake"       
    else:
        result="Profile Not found"
        found="notfound" 
    
    return JsonResponse({"success":1,"message":result,"found":found});     
   except Exception as e:
     return JsonResponse({"success":0,"message":str(e),"found":-1});   
    