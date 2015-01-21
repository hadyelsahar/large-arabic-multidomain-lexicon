# -*- coding: utf-8 -*-

import os 
import argparse
import csv

import numpy as np
import pandas as pd

from nltk.util import ngrams
from nltk.tokenize import TreebankWordTokenizer

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.grid_search import GridSearchCV

from classes.Document import *
from classes.Utils import * 
from classes.DeltaTfidf import * 


## Calculating Docsize in words, Doc richness with pos, neg words
## Predicted class using different classifiers
## original class 
###Setup of the Experiment is 
## Features : TFIDF, Count, Delta-TFIDF
## classifier : SVM GridSearch C = [0.01,0.01,0.1,1,10,100,1000] 
## 80/20 Split Balanced 2C


valid_datasets = ["SUQ","TRH","MOV","LABR","RES"]

parser = argparse.ArgumentParser(description='Feature Selection Experiments')
parser.add_argument('-d','--dataset',help='which dataset to run experiment on',required=False)
parser.add_argument('-o','--output', help='ouput folder name',required=True)

args = parser.parse_args()
if args.dataset is None :
    datasets = valid_datasets
else :
    if args.dataset in valid_datasets:
        datasets = [args.dataset]
    else : 
        print " only available datasets are " + str(valid_datasets)
        sys.exit()

#Reading lexicon
lexicon = pd.read_csv("lexicon/ALL_lex.csv",encoding="utf-8")
poslex = list(lexicon["ngram"][lexicon.polarity > 0])
neglex = list(lexicon["ngram"][lexicon.polarity < 0])

negators = [i.decode("utf-8") for i in ["لا","لم","ليس","ليست","مش","لن","مو","موب","غير","لا يوحد","عدم","مش","ليسوا","ما","تراجع","معدومة","معدوم","مفيش","مافي","ما في","مافيش","ما فيش","يخلو من","مهو","مب","مافيه","مافيه","مابه","موفيه","مو فيه","ما به","ولا يوجد به اي","ولا","وغير","ومش","وليس","لا شيء","لا يوجد شئ","ما اشوف فيه شي","مااشوف فيه شي","والله منا","و الله منا"]]
postnegators = [i.decode("utf-8") for i in ["معدومة","معدوم","متدني","متدنيه","غير موجود","غير موجوده"]]

poslex+= [n+" "+l for l in neglex for n in negators]  #adding neg lexicon negated
poslex+= [l+" "+n for l in neglex for n in postnegators]  #adding neg lexicon negated

neglex+= [n+" "+l for l in poslex for n in negators]  #adding neg lexicon negated
neglex+= [l+" "+n for l in poslex for n in postnegators]  #adding neg lexicon negated


vectorizers  = {
                # "tfidf" : TfidfVectorizer(
                #         tokenizer=TreebankWordTokenizer().tokenize,
                #         ngram_range=(1,2),norm="l1",
                #         preprocessor = Document.preprocess
                #     ),
                "count" : CountVectorizer(
                        tokenizer=TreebankWordTokenizer().tokenize,
                        ngram_range=(1,2),
                        preprocessor = Document.preprocess
                    ),
                # "lex-domain" : LexiconVectorizer(
                #         lexfile='lexicon/%s_lex.csv'%dname,
                #         polarity = True,
                #         weightedcount = True,
                #         preprocessor = Document.preprocess
                #     ),
                # "lex-all" : LexiconVectorizer(
                #         lexfile='lexicon/ALL_lex.csv',
                #         polarity = True,
                #         weightedcount = True,
                #         preprocessor = Document.preprocess
                #     ),
                # "delta-tfidf" : DeltaTfidf(                            
                #         tokenizer = TreebankWordTokenizer().tokenize,
                #         preprocessor = Document.preprocess
                #     )
}


classifiers = {
            # "svm": LinearSVC(penalty="l1", dual=False),
            "svm_cv": GridSearchCV(
                LinearSVC(penalty="l1", dual=False),
                [{'C': [0.0001, 0.001, 0.1, 1, 10]}] #range of C coefficients to try
                # [{'C': [0.01]}] #range of C coefficients to try
                )
            # "LREG": LogisticRegression(penalty="l1", dual=False),
            # "BernoulliNB" : BernoulliNB(alpha=.01),                
            # "SGD" : SGDClassifier(loss="hinge", penalty="l1"),
            # "KNN" : KNeighborsClassifier(n_neighbors=5, algorithm='auto')
}


for vectorizer_name,vectorizer in vectorizers.items():
    for classifier_name, classifier in classifiers.items():
        print " Experimenting classifier: %s vectorizer: %s"%(classifier_name, vectorizer_name)

        #file for each heatmap 
        fname = os.path.join(args.output,"%s_%s_doclength_accuracy.csv"%(classifier_name, vectorizer_name))
        csvfile = open(fname, 'w')

        #Dataframe of Results
        classification_results = pd.DataFrame(columns=( 'dname', 'docid', 
                                                'wordcount', 'poscount', 
                                                'negcount','predicted',
                                                'trueclass'))

        for dname in datasets:

            (X_train,y_train,X_test,y_test) = create_dataset(dname, CV = False, neutral = False, 
                        balanced = True, n_folds = 5 )[0]   #assume no cross validation will be selected

            #return hash of tokens in document
            def hash(doc):
                tokens = TreebankWordTokenizer().tokenize(doc)
                l1 = [" ".join(i) for i in ngrams(tokens,1)]
                l2 = [" ".join(i) for i in ngrams(tokens,2)]
                return set(l1 + l2)


            countpos = lambda doc : sum([1 for i in poslex if i in doc])
            countneg = lambda doc : sum([1 for i in neglex if i in doc])
            countword = lambda doc : len(TreebankWordTokenizer().tokenize(doc))

            X_test_proc = X_test.map(Document.preprocess)
            X_wordcount = X_test_proc.map(countword)
            X_hash = X_test_proc.map(hash)
            X_poscount = X_hash.map(countpos)
            X_negcount = X_hash.map(countneg)
            X_poscount = X_hash.map(lambda l : len(l & set(poslex)))
            X_negcount = X_hash.map(lambda l : len(l & set(neglex)))

            X_train = vectorizer.fit_transform(X_train, y_train)
            X_test = vectorizer.transform(X_test)

            classifier.fit(X_train,y_train)
            y_pred = classifier.best_estimator_.predict(X_test)


            temp = pd.DataFrame({
                'dname' : [dname for i in range(X_test.shape[0])], 
                'docid' : [i for i in range(X_test.shape[0])], #arbitary id dname_id
                'wordcount' : X_wordcount,
                'poscount'  : X_poscount,
                'negcount' : X_negcount,
                'predicted' : y_pred ,
                'trueclass' : y_test
                })

            classification_results = pd.concat([classification_results,temp])

        classification_results.to_csv(csvfile,index = False, index_label = False, escapechar = "\\")
        csvfile.close()

                










