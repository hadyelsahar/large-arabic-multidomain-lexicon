# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import StratifiedKFold, train_test_split



def create_dataset(dname, balanced = False, CV = False, neutral = False, n_folds = 5):

    docs = pd.read_csv('datasets/%s.csv'%dname,encoding="utf-8")
    #removing missing data
    docs = docs[docs["polarity"].notnull() & docs["text"].notnull()] 
    
    posdocs = docs[docs["polarity"] > 0]
    negdocs = docs[docs["polarity"] < 0]
    neudocs = docs[docs["polarity"] == 0]
    poslen = posdocs.shape[0]
    neglen = negdocs.shape[0]
    neulen = neudocs.shape[0]



    if not neutral:
        docs = docs[docs["polarity"] != 0]
        minlen = minlen = min([poslen, neglen])
    else : 
        minlen = min([poslen, neglen, neulen])


    #some Datasets doesn't have Neutrals QYM 
    if len(docs.polarity.unique()) is 2 :
        if balanced :            
            docs = pd.concat([posdocs[0:minlen], negdocs[0:minlen]])
    else :     
        if balanced :            
            docs = pd.concat([posdocs[0:minlen], 
                            negdocs[0:minlen], neudocs[0:minlen]])
                
    if CV :
        docs["polarity"]
        kf = StratifiedKFold(list(docs["polarity"]), 
                n_folds = n_folds, shuffle = False, random_state = 2)

    else : 
        (trainids,testids) = train_test_split(
                            range(len(docs.index)),
                            train_size = 0.8,
                            test_size = 0.2,random_state = 2)

        kf = [(trainids,testids)]



    kfolds_data = [] 


    # expanding data to arrays is better than using iterators
    # as we might loop over the same datasets many times
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

        kfolds_data.append((X_train,y_train,X_test,y_test))


    return kfolds_data
