# -*- coding: utf-8 -*-
import operator 
import csv 

import numpy as np
from scipy import arange

import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 

from nltk.tokenize import TreebankWordTokenizer
from classes.Document import *


datasets  = ["QYM","LABR","MOV","SUQ","TRR","TRH"]

classifiers = { 'LSVC' : LinearSVC,
                'LREG' : LogisticRegression
              }

#modes = {1:"run",2:"test",3:"fixed cr value"}  #modes for picking c 
mode = 1
cr = [20]

# Reading Datasets
for dataset_name in datasets:
    for classifier_name,classifier_class in classifiers.items():
        print " Experimenting dataset " + dataset_name

        # Reading datasets
        docs = pd.read_csv('./datasets/'+dataset_name+'.csv',
                    delimiter=",",encoding="utf-8")

        # Splitting dataset to Train and Test
        (trainids,testids) = cross_validation.train_test_split(
                                docs.index,
                                train_size=0.8,
                                test_size = 0.2,random_state=2)
        
        docs["text"] = docs["text"].map(lambda x : Document.preprocess(x).encode("utf-8"))
        train = docs.loc[trainids]
        test = docs.loc[testids]

        # Building TFIDF Vectors X (vector of TFIDF weights)
        # y (polarity class of the vector)
        vectorizer = TfidfVectorizer(
                        tokenizer=TreebankWordTokenizer().tokenize,
                        ngram_range=(1,2),norm="l1")

        Xtrain = vectorizer.fit_transform(train["text"])    
        ytrain = train["polarity"]
        Xtest = vectorizer.transform(test["text"])
        ytest= test["polarity"]

        fn = vectorizer.get_feature_names() # feature names
        acc = []                            # accuracy
        fc = []                  # feature counts for each iteration
        fs = []                  # feature ids for each iteration


        if mode is 1 :         
            tmp = [pow(10,i) for i in range(-2,7,1)]
            cr = [i+(x*i) for i in tmp for x in range(1,11)]
        elif mode is 2 :
            cr = [pow(10,i) for i in range(-2,4,1)]
        elif mode is 3 :
            # cr 
            pass
    
        for c in cr:
            try :

                svc = classifier_class(C=c, penalty="l1", dual=False)
                Xtrain_new = svc.fit_transform(Xtrain,ytrain)
                predicted = svc.predict(Xtest)
                
                accuracy = np.mean(predicted == ytest)               
                feature_count = Xtrain_new.shape[1]

                print "classifier : %s dataset : %s acc = %s at c = %s \
                and feat no. = %s and f_percent = %s" \
                %(classifier_name, dataset_name,accuracy,
                    c, feature_count,
                    str(float(feature_count)/Xtrain.shape[1])
                    )

                acc.append(accuracy)
                fc.append(feature_count)

                tmp = [(i,v) for i,v in enumerate(svc.coef_[0]) if v != 0 ]
                fs.append(dict(tmp))

            except ValueError, e:
                print e 
                print "at classifier : %s dataset : %s at c = %s "\
                        %(classifier_name, dataset_name, c)
                pass
        
        # logging results to files
        prefix = "./experiments-results/"
        f_bestfeatures = open(prefix + \
            classifier_name + "_" + dataset_name + "_bestfeatures.csv","w")
        f_acc = open(prefix + classifier_name + "_" + \
            dataset_name + "_nf_vs_acc.csv","w")

        f_acc.write("features_number,accuracy,feature_precentage")    
        for i,c in enumerate(acc):
            perc = float(fc[i])/Xtrain.shape[1]
            f_acc.write("\n%s,%s,%s"%(fc[i],c,perc))

        # get features for max accuracy 
        # if max accuracy get the minimum feature counts
        # acc is sorted according to increasing of number of features
        acc_r = [round(i,2) for i in acc]
        id_maxacc = acc_r.index(max(acc_r))
        best_features = fs[id_maxacc]
        best_features = sorted( best_features.items(), 
                                key=operator.itemgetter(1))

        
        fieldnames = ['ngram','weight','polarity']
        csvwriter = csv.writer(f_bestfeatures, delimiter=',', 
        	quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(fieldnames)
        for i,w in best_features:
            ngram = fn[i].encode("utf-8")
            cls = 1 if w > 0 else -1
            csvwriter.writerow([ngram,w,cls])

        f_bestfeatures.close()
        f_acc.close()

