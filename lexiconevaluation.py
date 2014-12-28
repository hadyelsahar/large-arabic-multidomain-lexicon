# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
#dataset prepataion 

datasets = ["LABR","SUQ","QYM","TRH","TRA","TRR","MOV","ALL","ALL_S","ALL_L"]
datasets = ["SUQ"]
#Building Features 

#vectorizers : 
#TFIDF
tfidf_vectorizer = TfidfVectorizer(
                        tokenizer=TreebankWordTokenizer().tokenize,
                        ngram_range=(1,2),norm="l1")
#Count
count_vectorizer = CountVectorizer()


#Feature Building
features = FeatureUnion([
    # ("count", count_vectorizer), 
    ("tfidf", tfidf_vectorizer)])


#Feature Selecting 
selector = VarianceThreshold()

#Classifiers 
svc = LinearSVC(penalty="l1", dual=False, C= 10)


#Pipeline 
classifier = Pipeline([
    ('features', features), 
    # ('select',selector), 
    ('svc', svc)])



for dname in datasets : 
    
    docs = pd.read_csv('datasets/%s.csv'%dname,encoding="utf-8")

    (trainids,testids) = cross_validation.train_test_split(
                            docs.index,
                            train_size=0.8,
                            test_size = 0.2,random_state=2)
        
    # docs["text"] = docs["text"].map(lambda x : Document.preprocess(x).encode("utf-8"))
    train = docs.loc[trainids]
    test = docs.loc[testids]


    classifier.fit_transform(train["text"],train["polarity"])

    pred = classifier.predict(test["text"])
  
    print metrics.classification_report(test["polarity"], pred,
        target_names=["neg","pos"])
