# -*- coding: utf-8 -*-

import pandas as pd
from nltk.util import ngrams
from nltk.tokenize import TreebankWordTokenizer
from classes.Document import * 
import numpy as np

### Calculating/ visualization of Richness of various datasets 
### with Sentimentwords
### should run script from inside visualization folder 

valid_datasets = ["SUQ","QYM","TRH","TRR","MOV","LABR","RES"]

lexicon = pd.read_csv("lexicon/ALL_lex.csv",encoding="utf-8")
poslex = lexicon["ngram"][lexicon.polarity > 0]
neglex = lexicon["ngram"][lexicon.polarity < 0]

dstat = {}

for dname in valid_datasets:
	
    docs = pd.read_csv("datasets/%s.csv"%dname,encoding="utf-8")

    docs["text"] = docs["text"].map(Document.preprocess)

    def hash(doc):
        tokens = TreebankWordTokenizer().tokenize(doc)
        l1 = [" ".join(i) for i in ngrams(tokens,1)]
        l2 = [" ".join(i) for i in ngrams(tokens,2)]
        return set(l1 + l2)    

    docs["hash"] = docs["text"].map(hash)

    countpos = lambda doc : sum([1 for i in poslex if i in doc])
    countneg = lambda doc : sum([1 for i in neglex if i in doc])

    docs["poscount"] = docs["hash"].map(countpos)
    docs["negcount"] = docs["hash"].map(countneg)


    #display
    from collections import OrderedDict
    # we throw the data into a pandas df
    from bokeh.charts import Bar

    grp = docs.groupby("polarity").mean()
    
    dstat[dname] = grp

    # later, we build a dict containing the grouped data
    


["negative","neutral","positive"]
posavglist = []
negavglist = []
namelist = [] 
for  k,v in dstat.items():
    posavglist.append(list(v.poscount))
    negavglist.append(list(v.negcount))
    namelist.append([k+"_neg",k+"_neu",k+"_pos"])

    
counts = OrderedDict(poscount=posavglist, negcount=negavglist)
bar = Bar(counts, namelist , filename="counts.html")

bar.title("Occurrence of positive and negative counts").xlabel("class").ylabel("count")
bar.legend(True).width(900).height(400).stacked(True)
bar.show()







