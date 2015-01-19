# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import *
import pandas as pd
import regex 
from sklearn.base import TransformerMixin
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.util import ngrams


class DeltaTfidf(TransformerMixin):
    """ Delta tfidf vectorizer transforming each document
    to a vector of delta-tfidf scores
    . more details for delta-tfidf implementation : http://bit.ly/1IlbenW
    . this implementation doesn't assume balanced datasets 
    uses the formula in the paper before balanced assumption 
    . this implementation only 
    """
    
    def __init__(self, tokenizer, preprocessor=None):
        """
        """
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.posvectorizer = None
        self.negvectorizer = None

    def transform(self, X, **transform_params):
        """
        overriding transform function 
        """

        deltatfidf = self.posvectorizer.transform(X) - \
                        self.negvectorizer.transform(X)
        return deltatfidf
        
    def fit(self, X, y, **fit_params):
        v = []
        for doc in X:
            doc = self.preprocessor(doc)
            u = self.tokenizer(doc)
            b = [" ".join(i) for i in ngrams(u,2)]
            v += u
            v += b 
        

        #unique all unigrams and bigrams
        v = [i for i in v if len(i.strip()) > 0]
        vocab = set(v)

        pos = [val for i,val in enumerate(X) if list(y)[i] == 1]
        neg = [val for i,val in enumerate(X) if list(y)[i] == -1]    

        self.posvectorizer = TfidfVectorizer(
                            tokenizer=self.tokenizer,
                            ngram_range=(1,2),
                            preprocessor = self.preprocessor,
                            vocabulary = vocab,
                            norm = "l1"
                        ).fit(pos)

        self.negvectorizer = TfidfVectorizer(
                            tokenizer=self.tokenizer,
                            ngram_range=(1,2),
                            preprocessor = self.preprocessor,
                            vocabulary = vocab,
                            norm = "l1"
                        ).fit(neg)

        return self


    def get_feature_names(self):
        return self.posvectorizer.get_feature_names()





