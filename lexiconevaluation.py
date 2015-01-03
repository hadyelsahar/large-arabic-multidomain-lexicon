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

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classes.Document import *
from classes.LexiconVectorizer import * 


def create_dataset(dname, balanced = False, CV = False, neutral = False):

    docs = pd.read_csv('datasets/%s.csv'%dname,encoding="utf-8")

    posdocs = docs[docs["polarity"] > 0]
    negdocs = docs[docs["polarity"] < 0]
    neudocs = docs[docs["polarity"] == 0]
    poslen = posdocs.shape[0]
    neglen = negdocs.shape[0]
    neulen = neudocs.shape[0]

    minlen = min([poslen, neglen, neulen])

    #tar_docs

    if balanced :
        docs = pd.concat([posdocs[0:minlen], 
                        negdocs[0:minlen], neudocs[0:minlen]])

    if not neutral:
        docs = docs[docs["polarity"] != 0]

    if CV :
        kf = StratifiedKFold(docs["polarity"], n_folds=5)

    else : 
        (trainids,testids) = cross_validation.train_test_split(
                            docs.index,
                            train_size=0.8,
                            test_size = 0.2,random_state=2)

        kf = [(trainids,testids)]

    for trainids, testids in kf:
        
        #convert position in array sent to CV object to index in Dataframe
        trainids = [docs.index[i] for i in trainids]
        testids = [docs.index[i] for i in testids]
        train = docs.loc[trainids]
        test = docs.loc[testids]

        X_train = train["text"]
        y_train = train["polarity"]
        X_test = test["text"]
        y_test = test["polarity"]

        yield  (X_train,y_train,X_test,y_test)


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
    ("count", count_vectorizer),
    ("bool_lex", lex_vectorizer),
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

for dname in datasets : 
    
    for i in create_dataset(dname, CV = True, neutral = True, balanced = False):

        (X_train,y_train,X_test,y_test) = i

        classifier.fit_transform(X_train,y_train)
        pred = classifier.predict(X_test)
        arr.append(precision_recall_fscore_support(y_test, pred))

    metrics = [sum([i[x] for i in arr])/len(arr) for x in range(4)]
    
    x = pd.DataFrame(metrics).transpose()
    x.index = ["neg","neutral","pos"] if len(metrics[0]) == 3 else ["neg","pos"]
    x.columns = ["precision","recall","fscore","support"]

    print x 








