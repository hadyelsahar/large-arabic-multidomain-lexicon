#encoding:utf-8 
'''
Created on Oct, 2014
python 2.7.3 
@author: hadyelsahar 
'''

import ConfigParser
import pandas as pd
import codecs
import regex
import re


# Data Structure to represent document words
class Document:

    @staticmethod
    def preprocess(text):
                
        text = Document.remove_elongation(text)
        text = Document.normalize(text)
        text = Document.clean(text)
        # text = Document.tag(text)

        return text

    @staticmethod
    def remove_elongation(text):        

        return regex.sub(r'(.)\1{3,}',r'\1\1', text, flags=regex.UNICODE)

    @staticmethod
    def normalize(text):    

        normLetters = {"إ":"ا","آ":"ا","ﻹ":"ﻻ","ﻷ":"ﻻ","ﻵ":"ﻻ"}
        normnumbers = {"٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9"}
        normpunc = {"،":",","؟":"?"}

        normLetters  =  dict(normnumbers.items() + normLetters.items() + normpunc.items())
        normText = text
        for  k in normLetters.keys():
            w = k.decode("utf-8")
            R = normLetters[k].decode("utf-8")
            if w in normText:
                normText = normText.replace(w,R)
        
        return normText

    @staticmethod
    def clean(text):
                
        #remove tashkeel 
        tashkeel = r"[ًٌٍَُِّ~ْ:؛×÷`]"
        text = regex.sub(tashkeel,"", text, flags=regex.UNICODE)

        r = u"أابتثجحخدذرزسشصضطظعغفقكلمنهةويىﻻؤئءabcdefghijklmnopqrstuvwxyz1234567890=\\/-+!.:><*%&:?)($,\"' "
        smileys = u"😍❤😘😖😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟😠😡😢😣😥😦😧😨😩😪😫😭😮😯😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿🙀"

        r = r+smileys

        for l in [t for t in text if t not in r]: 
            text = text.replace(l," ")

        punc = ".?!,"

        #separating punc from original words
        for p in punc:
            text = text.replace(p," "+p+" ")
        
        #removing extra spaces
        text = regex.sub(r'[\s\n]+',' ', text, flags=regex.UNICODE)

        return text


    @staticmethod
    def tag(text):
        dic = pd.read_csv("/home/hadyelsahar/Dropbox/Nile university/Research/large-arabic-multidomain-lexicon/manual-annotation/QYM/QYM_bestfeatures_filter2.csv",encoding="utf-8")
        
        posw = dic[dic.polarity == 1]["ngram"]
        negw = dic[dic.polarity == -1]["ngram"]

        for w in posw:
            text = text.replace(" "+w+" ",u" POS__TAG ")

        for w in negw:
            text = text.replace(" "+w+" ",u" NEG__TAG ")

        neg = ["لا","لم","ليس","ليست","مش","لن","مو","موب","غير","لا يوحد","عدم","مش","ليسوا","ما","تراجع","معدومة","معدوم","مفيش","مافي","ما في","مافيش","ما فيش","يخلو من","مهو","مب","مافيه","مافيه","مابه","موفيه","مو فيه","ما به","ولا يوجد به اي","ولا","وغير","ومش","وليس","لا شيء","لا يوجد شئ","ما اشوف فيه شي","مااشوف فيه شي","والله منا","و الله منا"]
        pneg = ["معدومة","معدوم","متدني","متدنيه","غير موجود","غير موجوده"]
        ints = ["كبير","كبير جدا","قوي","جدا","اوى","أوى","فشخ","فحت","طحن","خالص","بعنف","بقوه","بشده","بغباوه","تماما","السنين","فحت ردم","نيك","خالص","مرره","جداا","بشدده","قووي"]
        
        for w in neg:
            text = text.replace(" "+w.decode("utf-8")+" ",u" NEGA__TOR ")

        for w in pneg:
            text = text.replace(" "+w.decode("utf-8")+" ",u" POST__NEGATOR ")

        for w in ints:
            text = text.replace(" "+w.decode("utf-8")+" ",u" INTEN__SIFIER ")

        return text