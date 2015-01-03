# -*- coding: utf-8 -*-
import pandas as pd 

docs = pd.read_csv('./output.csv',encoding="utf-8").drop_duplicates()
docs['polarity'] = docs.rating.map(lambda x :1 if x > 3 else -1 if x < 3 else 0)
docs["text"] = docs["otherText"]

pred = lambda obj: obj['polarity'].nunique() == 1
docs = docs.groupby('text').filter(pred).drop_duplicates('text')

docs = docs[docs['polarity'].notnull()]
docs.to_csv("../../../datasets/SUQ.csv",columns=["text","polarity"],encoding="utf-8",index=False)
