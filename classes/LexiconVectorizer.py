# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import regex 
from sklearn.base import TransformerMixin


class LexiconVectorizer(TransformerMixin):
    """ Custom Tranformer class to transform a vectorize 
    set of documents this class supports creating vectors 
    for each document based on a given lexicon
    """
    
    def __init__(self, lexfile, polarity = True, weightedcount = False,
        preprocessor=None):
        """
        lexfile : is a given file contains lexicon to be used in csv format 
        with column names
            . "ngram" for dictionary words
            . "polarity" for polarity class

        polarity : 
            if False : 
                feature value will be only existence of leixcon words 
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
        self.methodregex = methodregex

    def transform(self, X, **transform_params):

        features = []
        for doc in X : 
            vector = []
            if self.preprocessor is not None:
                doc = self.preprocessor(doc)

            for w in self.lexicon["ngram"]:
                #regex takes more time
                #few amount of lexicon terms exists in each doc
                
                p = int(self.lexicon["polarity"][self.lexicon["ngram"] == w])
                if w in doc:
                    r = regex.compile(r"(?:^|\s+)%s(?:\s+|$)" % regex.escape(w) ,encoding="utf-8")
                    n = len(r.findall(doc))

                else :
                    n = 0 

                if  n > 0:
                    if self.polarity:
                        if self.weightedcount:
                            v = n * p 
                        else :
                            v = p                        
                    else : 
                        v = 1 
                else :
                    v = 0 

                vector.append(v)
            features.append(vector)
                
        return features
                
        
    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names():
        return self.lexicon["ngrams"]





