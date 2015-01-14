# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import pandas as pd
import regex 
from sklearn.base import TransformerMixin
from nltk.util import ngrams
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import * 

class LexiconVectorizer(TransformerMixin):
    """ Custom Tranformer class to transform a vectorize 
    set of documents this class supports creating vectors 
    for each document based on a given lexicon
    """
    
    def __init__(self, lexfile, polarity = True, weightedcount = False,
        preprocessor=None):
        """
        lexfile : Is a given file contains lexicon to be used in csv format 
        with column names
            . "ngram" for dictionary words
            . "polarity" for polarity class

        polarity : 
            if False : 
                Feature value will be only existence of leixcon words 
                0 if exists and 1 if not 
            if True : count feature values will be 0 for non existence 
                and lexiconword polarity [1,-1] if it exists

        weightedcount : if a lexicon word appears "n" times in a document,
            feature value will be n * p , where p is the word polarity/weight 
            in the lexicon file
        """

        self.polarity = polarity
        self.weightedcount = weightedcount
        self.preprocessor = preprocessor
        
        lex = pd.read_csv(lexfile,encoding="utf-8")
        pred = lambda obj: obj['polarity'].nunique() == 1
        lex = lex.groupby("ngram").filter(pred).drop_duplicates("ngram")
        self.lexicon = lex

    def transform(self, X, **transform_params):

        #sparse matrix with occurrences nxm
        # n : number of docs
        # m : size of lexicon 
        features = np.empty((len(X),len(self.lexicon)))            

        for docid,doc in enumerate(X):
            if self.preprocessor is not None:
                doc = self.preprocessor(doc)

            tokens = TreebankWordTokenizer().tokenize(doc)
            bigrams = [" ".join(i) for i in ngrams(tokens,2)]
            doctokens = tokens + bigrams
            
            tokencounts = Counter(doctokens)            
            match = set(tokencounts.keys()) & set(self.lexicon["ngram"])

            if len(match) > 0 :
                #occurrences vector
                occurrences = self.lexicon["ngram"].map(lambda w : w in match)
                ovec = csr_matrix(occurrences)
                #polarity vector
                pvec = csr_matrix(self.lexicon["polarity"])
                #counts vector
                counts = self.lexicon["ngram"].map(lambda w : tokencounts[w] if w in match else 0 )
                cvec = csr_matrix(counts)
                
                if self.polarity:
                    if self.weightedcount:
                        vector = ovec.multiply(pvec).multiply(cvec)
                    else :
                        vector = ovec.multiply(pvec)
                else : 
                    if self.weightedcount:
                        vector = ovec.multiply(cvec)
                    else :
                        vector = ovec         
                vector = vector.todense()
            else :
                #can't skip because np.empty is > 0 
                vector = np.zeros(len(self.lexicon))
        
            features[docid] = vector

        return csr_matrix(features)
                
        
    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names():
        return self.lexicon["ngrams"]





