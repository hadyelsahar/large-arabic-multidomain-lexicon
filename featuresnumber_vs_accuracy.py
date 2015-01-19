# -*- coding: utf-8 -*-
import operator 
import os
import csv 
import argparse

import numpy as np
from scipy import arange

import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

from nltk.tokenize import TreebankWordTokenizer

from classes.Document import *
from classes.Utils import * 
from classes.DeltaTfidf import * 


valid_datasets = ["SUQ","QYM","TRH","TRR","MOV","LABR","RES"]
valid_datasets = ["SUQ","TRH","MOV","LABR","RES"]

parser = argparse.ArgumentParser(description='Feature Selection Experiments')
parser.add_argument('-d','--dataset',help='which dataset to run experiment on',required=False)
parser.add_argument('-o','--output', help='ouput file name',required=True)

args = parser.parse_args()
if args.dataset is None :
    datasets = valid_datasets
else :
    if args.dataset in valid_datasets:
        datasets = [args.dataset]
    else : 
        print " only available datasets are " + str(valid_datasets)
        sys.exit()


#modes = {1:"run",2:"test",3:"fixed cr value"}  #modes for picking c 
mode = 2
fixed_C_value = 10

if mode is 1 :         
    tmp = [pow(10,i) for i in range(-2,7,1)]
    C_range = [i+(x*i) for i in tmp for x in range(1,11)]
elif mode is 2 :
    C_range = [pow(10,i) for i in range(-2,2,1)]
elif mode is 3 :
    C_range = [fixed_C_value] 


classifiers = { 
                # 'LREG' : LogisticRegression,
                "SVM" : LinearSVC(penalty="l1", dual=False)                
              }

vectorizers  = {
                "tfidf" : TfidfVectorizer(
                        tokenizer=TreebankWordTokenizer().tokenize,
                        ngram_range=(1,2),norm="l1",
                        preprocessor = Document.preprocess
                    ),
                "count" : CountVectorizer(
                        tokenizer=TreebankWordTokenizer().tokenize,
                        ngram_range=(1,2),
                        preprocessor = Document.preprocess
                    ),
                "delta-tfidf" : DeltaTfidf(                            
                        tokenizer = TreebankWordTokenizer().tokenize,
                        preprocessor = Document.preprocess
                    )
}


csvfile = open(args.output, 'w')
writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(["dataset","vectorizer","c","featurescount","featurespercentage","score"])

for vectorizer_name,vectorizer in vectorizers.items():
    for dname in datasets:
        for classifier_name,classifier in classifiers.items():
            print " Experimenting dataset: %s vectorizer: %s"%(dname, vectorizer_name)

            for c in C_range :                         
            
                (X_train,y_train,X_test,y_test) = create_dataset(dname, 
                        CV = False, neutral = False, balanced = False, n_folds = 5
                        )[0]

                X = vectorizer.fit_transform(X_train,y_train)
                y = y_train

                classifier.set_params(C=c)
                classifier.fit(X, y)
                y_pred = classifier.predict(vectorizer.transform(X_test))

                features_c = classifier.coef_.nonzero()[0].shape[0]
                features_perc = float(features_c )/classifier.coef_.shape[1]
                score = accuracy_score(y_test, y_pred)
            
                writer.writerow([dname,vectorizer_name,c,features_c,features_perc,score])            

csvfile.close()