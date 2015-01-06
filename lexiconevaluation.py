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

parser = argparse.ArgumentParser(description='script for designing experiments sentiment classification upon created datasets and generated lexicons')
parser.add_argument('-d','--dataset', help='which dataset to run experiment on',required=False)
args = parser.parse_args()


valid_datasets = ["LABR","SUQ","QYM","TRH","TRA","TRR","MOV"]
if args.dataset is None : 
    datasets = valid_datasets
else :
    if args.dataset in valid_datasets:
        datasets = [args.dataset]
    else : 
        print " only available datasets are " + str(valid_datasets)
        sys.exit()


# c = {"TRH":100,"QYM":30}

print datasets

for dname in datasets : 

    #Building Features 

    #vectorizers : 
    #TFIDF
    tfidf_vectorizer = TfidfVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1,2),norm="l1",
                            preprocessor = Document.preprocess
                            )

    #Count
    count_vectorizer = CountVectorizer()

    #Lexicon Vectorizers
    lex_vectorizer = LexiconVectorizer(lexfile='lexicon/%s_lex.csv'%dname,
                        polarity = True,
                        weightedcount = True,
                        preprocessor = Document.preprocess
                        )

    lex_all_vectorizer = LexiconVectorizer(lexfile='lexicon/ALL_lex.csv',
                        polarity = True,
                        weightedcount = True,
                        preprocessor = Document.preprocess
                        )


    #Feature Building
    features = FeatureUnion([
        # ("count", count_vectorizer),
        ("bool_lex", lex_all_vectorizer),
        ("tfidf", tfidf_vectorizer)
        ])

    #Feature Selecting 
    # selector = VarianceThreshold()

    #Classifiers
    # svc = LinearSVC(penalty="l1", dual=False, C= c[dname])
    svc = LinearSVC(penalty="l1", dual=False)

    #Pipeline 
    classifier = Pipeline([
        ('features', features), 
        # ('select',selector), 
        ('svc', svc)])

    
    #dataset prepataion 
    fold = create_dataset(dname, CV = True, 
                    neutral = True, balanced = True, n_folds = 5)
    arr = []
    for (X_train,y_train,X_test,y_test)  in fold:

        classifier.fit_transform(X_train,y_train)
        pred = classifier.predict(X_test)

        a1 = np.array(precision_recall_fscore_support(y_test, pred))
        a2 = np.array(precision_recall_fscore_support(y_test, pred, average = "micro"))
        a2[3] = np.sum(a1[3,:]) #adding support count = sum of all supports
        arr.append(np.c_[a1,a2])    


    metrics = np.array([sum([i[x] for i in arr])/len(arr) for x in range(4)])
    x = pd.DataFrame(metrics).transpose()
    x.index = ["neg","neutral","pos","Average/Total"] if len(metrics[0]) == 4 \
                else ["neg","pos","Average/Total"]
    x.columns = ["precision","recall","fscore","support"]
    print x 

