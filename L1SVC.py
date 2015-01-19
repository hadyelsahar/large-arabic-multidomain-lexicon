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
from nltk.tokenize import TreebankWordTokenizer
from classes.Document import *
from classes.Utils import * 
from classes.DeltaTfidf import * 


valid_datasets = ["SUQ","QYM","TRH","TRR","MOV","LABR","RES"]
valid_datasets = ["SUQ","TRH","MOV","LABR","RES"]

parser = argparse.ArgumentParser(description='Feature Selection Experiments')
parser.add_argument('-d','--dataset',help='which dataset to run experiment on',required=False)
parser.add_argument('-o','--output', help='ouput ""directory"" name',required=True)

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
                "svm_cv": GridSearchCV(
                    LinearSVC(penalty="l1", dual=False),
                    [{'C': C_range}],
                    cv = 5
                )
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


for vectorizer_name,vectorizer in vectorizers.items():
    for dname in datasets:
        for classifier_name,classifier in classifiers.items():
                        

            print " Experimenting dataset: %s vectorizer: %s"%(dname, vectorizer_name)

            (X_train,y_train,X_test,y_test) = create_dataset(dname, 
                    CV = False, neutral = False, balanced = False, n_folds = 5
                    )[0]

            X = vectorizer.fit_transform(X_train,y_train)
            y = y_train


            classifier.fit(X, y)

            coeffs = classifier.best_estimator_.coef_
            feature_counts = coeffs.nonzero()[0].shape[0]
            fn = [i.encode("utf-8") for i in vectorizer.get_feature_names()]
            sfn = np.array(fn,dtype=np.str_)[coeffs.nonzero()[1]]
            sfv = np.array(coeffs[coeffs.nonzero()],dtype=np.float32)
            sfp = [1 if i > 0 else -1 for i in sfv]
            selected_features = np.array([sfn,sfv,sfp])
            # sorting feature vector according to coefficient values 
            selected_features = selected_features[:,sfv.argsort()]
        
            fname = os.path.join(args.output,"%s_%s.csv"%(dname, vectorizer_name))
            with open(fname, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["ngram","weight","polarity"])
                writer.writerows(selected_features.T)

            for i in classifier.grid_scores_:
                score = i.mean_validation_score
                c = i.parameters["C"]
                print "%s \t %s \t %s \t %s"%(dname, vectorizer_name, c, score)



            