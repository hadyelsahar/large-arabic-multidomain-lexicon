# -*- coding: utf-8 -*-
import pandas as pd 

docs = pd.read_csv('./elcinema.csv',encoding="utf-8")
docs['polarity'] = docs.score[docs.score !=6].map(lambda x :1 if x > 6 else -1)
docs = docs[["text","polarity"]][docs['polarity'].notnull()]
docs.to_csv("../../datasets/MOV.csv",encoding="utf-8")