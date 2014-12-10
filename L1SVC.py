# -*- coding: utf-8 -*-
import operator 

import numpy as np
from scipy import arange

import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import cross_validation
from sklearn.svm import LinearSVC

from nltk.tokenize import TreebankWordTokenizer


datasets  = ["QYM"]
#Reading Datasets 
for dataset_name in datasets:
    print " Experimenting dataset " + dataset_name

    # Reading datasets
    docs = pd.read_csv('./datasets/'+dataset_name+'.csv',
                delimiter=",",encoding="utf-8")

    # Splitting dataset to Train and Test
    (trainids,testids) = cross_validation.train_test_split(docs.index,
                            train_size=0.8,test_size = 0.2,random_state=2)
    train = docs.loc[trainids]
    test = docs.loc[testids]

    # Building TFIDF Vectors X (vector of TFIDF weights)
    # y (polarity class of the vector)

    vectorizer = TfidfVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
            ngram_range=(1,2))
    Xtrain = vectorizer.fit_transform(train.text)    
    ytrain = train.polarity
    Xtest = vectorizer.transform(test.text)
    ytest= test.polarity

    fn = vectorizer.get_feature_names() #feature names
    acc = []                            #accuracy
    fc = []                             #feature counts for each iteration
    fs = []                  #feature ids for each iteration

    # cr = [pow(10,i) for i in range(-5,5,1)]
    cr = arange(1.4,1.8,0.01)   
    for c in cr:
        try :
            svc = LinearSVC(C=c, penalty="l1", dual=False)
            Xtrain_new = svc.fit_transform(Xtrain,ytrain)
            predicted = svc.predict(Xtest)
            
            accuracy = np.mean(predicted == ytest)
            feature_count = Xtrain_new.shape[1]

            print "dataset : " + dataset_name + "acc = " + str(accuracy) + \
                    " at c = " + str(c)  + " and n_features = " + \
                    str(feature_count)

            acc.append(accuracy)
            fc.append(feature_count)
            fs.append(dict([(i,v) for i,v in enumerate(svc.coef_[0]) if v != 0 ]))

        except : 
            pass
    

    # logging results to files

    prefix = "./experiments-results/"
    f_bestfeatures = open(prefix + \
        "L1SVC_" + dataset_name + "_bestfeatures.csv","w")
    f_acc = open(prefix + "L1SVC_" + \
    	dataset_name + "_nf_vs_acc.csv","w")

    f_acc.write("features_number,accuracy")    
    for i,c in enumerate(acc):
        f_acc.write("\n%s,%s"%(fc[i],c))

    # get features for max accuracy #if max accuracy get the minimum feature counts
    # acc is sorted according to increasing of number of features (increasing of c)
    
    id_maxacc = acc.index(max(acc))
    best_features = fs[id_maxacc]
    best_features = sorted(best_features.items(), key=operator.itemgetter(1))
    
    s = "\n".join([fn[i].encode("utf-8") + "," + str(w) for i,w in best_features])
    f_bestfeatures.write("ngram,weight\n"+s)

    f_bestfeatures.close()
    f_acc.close()


    