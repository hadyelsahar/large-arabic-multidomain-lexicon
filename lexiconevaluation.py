# -*- coding: utf-8 -*-

import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classes.Document import *
from classes.Utils import * 
from classes.LexiconVectorizer import * 


valid_datasets = ["SUQ","QYM","TRH","TRR","MOV","LABR","RES"]

parser = argparse.ArgumentParser(description='script for designing experiments sentiment classification upon created datasets and generated lexicons')
parser.add_argument('-d','--dataset',
    help='which dataset to run experiment on',required=False)
args = parser.parse_args()
if args.dataset is None : 
    datasets = valid_datasets
else :
    if args.dataset in valid_datasets:
        datasets = [args.dataset]
    else : 
        print " only available datasets are " + str(valid_datasets)
        sys.exit()


# c = {"TRH":100,"QYM":30}

for dname in datasets : 

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
                    "lex-domain" : LexiconVectorizer(
                            lexfile='lexicon/%s_lex.csv'%dname,
                            polarity = True,
                            weightedcount = True,
                            preprocessor = Document.preprocess
                        ),
                    "lex-all" : LexiconVectorizer(
                            lexfile='lexicon/ALL_lex.csv',
                            polarity = True,
                            weightedcount = True,
                            preprocessor = Document.preprocess
                        )
    }

    kfolds = {                
                "CV_unBalanced_2C" : create_dataset(dname, 
                    CV = True, neutral = False, balanced = False, n_folds = 5
                    ),
                "CV_unBalanced_3C" : create_dataset(dname, 
                    CV = True, neutral = True, balanced = False, n_folds = 5
                    ),
                "CV_Balanced_2C" : create_dataset(dname, 
                    CV = True, neutral = False, balanced = True, n_folds = 5
                    ),
                "CV_Balanced_3C" : create_dataset(dname, 
                    CV = True, neutral = True, balanced = True, n_folds = 5
                    ),
                "split_unBalanced_2C" : create_dataset(dname, 
                    CV = False, neutral = False, balanced = False, n_folds = 5
                    )
    }


    classifiers = {
                "svc" : LinearSVC(penalty="l1", dual=False),
                # "svc_penalized" : LinearSVC(penalty="l1", dual=False, C= c[dname])
    }


    #Feature Building
    features = {
                # "lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"])]
                #         ),
                "lex-all" : FeatureUnion([
                        ("lex-all", vectorizers["lex-all"])]
                        ),
                # "tfidf" : FeatureUnion([
                #         ("tfidf", vectorizers["tfidf"])]
                #         ),
                # "count" : FeatureUnion([
                #         ("count", vectorizers["count"])]
                #         ),
                # "tfidf_lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"]),
                #         ("tfidf", vectorizers["tfidf"])]
                #         ),
                # "tfidf_lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"]),
                #         ("tfidf", vectorizers["tfidf"])]
                #         ),
                # "count_lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"]),
                #         ("count", vectorizers["count"])]
                #         ),
                # "count_lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"]),
                #         ("count", vectorizers["count"])]
                #         )
    }

    
    for fvector_name,fvector in features.items():
        for clf_name, clf in classifiers.items():
            for fold_name,fold in kfolds.items():

                print "# %s\t%s\t%s\t%s"%(dname, fold_name, fvector_name, clf_name)
                pipeline = Pipeline([
                                ('features', fvector), 
                                # ('select',selector), 
                                ('classifier', clf)])

                # for each of the cross folds 
                # calculate the metrics and store them in array
                arr = []                
                for (X_train,y_train,X_test,y_test)  in fold:

                    pipeline.fit_transform(X_train,y_train)
                    pred = pipeline.predict(X_test)

                    a1 = np.array(precision_recall_fscore_support(y_test, pred))
                    a2 = np.array(precision_recall_fscore_support(y_test, pred, 
                        average = "micro"))
                    a2[3] = np.sum(a1[3,:]) #support count = sum of all supports
                    arr.append(np.c_[a1,a2])    

                # average all the metrics of all cross folds
                metrics = np.mean(np.array(arr),axis = 0)                
                x = pd.DataFrame(metrics).transpose()

                ic3 = ["neg","neutral","pos","Average/Total"]
                ic2 = ["neg","pos","Average/Total"]
                x.index =  ic3 if len(metrics[0]) == 4 else ic2                 
                x.columns = ["precision","recall","fscore","support"]

                print x 
                print "\n"