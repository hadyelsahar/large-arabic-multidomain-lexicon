# -*- coding: utf-8 -*-

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

lexname = "TRH"
lex_vectorizer = LexiconVectorizer(lexfile='lexicon/%s_lex.csv'%lexname,
                    polarity = True,
                    weightedcount = True,
                    preprocessor = Document.preprocess
                    )

    #Feature Building
features = FeatureUnion([
    # ("count", count_vectorizer),
    # ("bool_lex", lex_vectorizer),
    ("tfidf", tfidf_vectorizer)
    ])


    #Feature Selecting 
# selector = VarianceThreshold()

    #Classifiers
svc = LinearSVC(penalty="l1", dual=False, C= 10)

    #Pipeline 
classifier = Pipeline([
    ('features', features), 
    # ('select',selector), 
    ('svc', svc)])

#dataset prepataion 

datasets = ["LABR","SUQ","QYM","TRH","TRA","TRR","MOV"]
# datasets = ["LABR","SUQ","QYM","TRH","MOV"]
datasets = ["SUQ"]
arr = []

for dname in datasets : 
    
    for i in create_dataset(dname, CV = True, neutral = True, balanced = False):

        (X_train,y_train,X_test,y_test) = i

        classifier.fit_transform(X_train,y_train)
        pred = classifier.predict(X_test)

        a1 = np.array(precision_recall_fscore_support(y_test, pred))
        a2 = np.array(precision_recall_fscore_support(y_test, pred, average = "micro"))
        
        a2[3] = np.sum(a1[3,:]) #adding support count = sum of all supports
        
        a = np.c_[a1,a2]

        arr.append(a)    

    metrics = np.array([sum([i[x] for i in arr])/len(arr) for x in range(4)])
    
    x = pd.DataFrame(metrics).transpose()

    x.index = ["neg","neutral","pos","Average/Total"] if len(metrics[0]) == 4+ else ["neg","pos","Average/Total"]
    x.columns = ["precision","recall","fscore","support"]

    print x 








