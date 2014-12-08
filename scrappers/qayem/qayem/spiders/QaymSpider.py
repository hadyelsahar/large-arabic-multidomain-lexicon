# -*- coding: utf-8 -*-
import re 
import json

import scrapy
from scrapy import log

from qayem.items import Review

key="sXvrQYgGGhBdZVyMptf3"

class QayemSpider(scrapy.Spider):
    """Spider to Scrap all Reviews from Qaym.com"""
    
    name = "qaymspider"    
    
    start_urls = ['http://api.qaym.com/0.1/countries/key='+key]
    allowed_domains = ['qaym.com']

    def parse(self, response):
        """Scrap all countries from Qaym API"""

        j = response.body
        cs = json.loads(j)       
        for v in cs:
            cid = v['country_id']
            url = "http://api.qaym.com/0.1/countries/"+cid+"/cities/key="+key
            yield scrapy.Request(url, callback=self.parse_city)

    def parse_city(self, response):
        """Scrap all countries from Qaym API"""
        j = response.body
        cs = json.loads(j)
        
        if cs:
            for v in cs:
                cid = v['city_id']
                url = "http://api.qaym.com/0.1/cities/"+cid+"/items/key="+key
                yield scrapy.Request(url, callback=self.parse_restaurant)

    def parse_restaurant(self, response):
        j = response.body
        rs = json.loads(j)

        if rs:
            for v in rs:
                review = Review()
                rid = v['item_id']
                review['restaurant_id'] = rid
                url = "http://api.qaym.com/0.1/items/"+rid+"/votes/key="+key

                yield scrapy.Request(url,
                    callback=self.parse_votes, 
                    meta = {'item':review}
                    )

    def parse_votes(self,response):
        j = response.body
        vs = json.loads(j)

        if vs: 
            review = response.meta['item']

            votes = {}

            for vo in vs.values():
                for vi in vo :
                    votes[vi["user_id"]] = vi["vote"]
            
            rid = review['restaurant_id']
            url = "http://api.qaym.com/0.1/items/"+rid+"/reviews/key="+key
            yield scrapy.Request(url,
                callback=self.parse_review,
                meta = {'item':review,'votes':votes})

    def parse_review(self,response):
        
        review = response.meta['item']
        votes = response.meta['votes']
        j = response.body
        rs = json.loads(j)

        if rs:
            for v in rs:
                uid = v['user_id']
                if uid in votes:
                    review['user_id'] = uid
                    review['text'] = self.cleantxt(v['title'] + "\n" + v['text'])
                    review['polarity'] = votes[uid]
                    yield review

    def cleantxt(s):
        c = s.encode("utf-8").replace("<br />"," ")        
        c = c.replace('\r','')	
        c = c.replace('\t',' ')
        c = re.sub(r'\s+\n','\n',c)
        c = re.sub(r'\n\s+','\n',c)
        c = re.sub(r'\n+','\n',c)
        c = re.sub(r'\s+',' ',c)
        c = c.strip()       
        return c


