# -*- coding: utf-8 -*-
import re 
import json
import regex 

import scrapy
from scrapy import log

from elcinema.items import Review


class ElcinemaSpider(scrapy.Spider):
    """Spider to Scrap all Reviews from http://www.elcinema.com/"""
    
    name = "elcinemaspider" 
    baseurl = "http://www.elcinema.com"   
    
    start_urls = ['http://www.elcinema.com/reviews/']
    allowed_domains = ['elcinema.com']

    def parse(self, response):
        """Scrap all available dates for reviews"""
        
        years = response.xpath('//select[@id="chooseYear"]/option/@value')
        for year in years.extract():
            months = response.xpath('//select[@id="chooseMonth"]/option/@value')
            for month in months.extract():
                url = "http://www.elcinema.com/reviews/%s/%s/"%(year,month)
                yield scrapy.Request(url, callback=self.parse_month)

    def parse_month(self,response):
        """scrap all reviews for all dates"""

        rlnks = response.xpath('//a[contains(@href,"/review/")]/@href')
        for u in set(rlnks.extract()) :
            url = self.baseurl+u
            yield scrapy.Request(url, callback=self.parse_review)

        nextlnk = response.xpath(u'//a[contains(text(),"التالي")]/@href').extract()
        if len(nextlnk) > 0 :        
            url = self.baseurl+nextlnk[0]
            yield scrapy.Request(url, callback=self.parse_month)
     
    def parse_review(self,response):
        review = Review()

        usr = response.css('.padded1-h')\
                .xpath('.//li/a[contains(@href,"/user/")]/@href')

        review['user_id'] = usr.extract()[0].replace("/review","")
        
        score = response.xpath('//li[contains(@title,"10]")]/@title')
        score = score.extract()[0].replace("[","")
        score = score.replace("/10]","").strip()
        review['score'] = int(score)

        title = response.css(".boxed-1")[0].xpath("./h3/text()").extract()[0].strip()
        body = "\n".join(response.css(".padded1-h")[0]\
                    .xpath('./div/div[contains(@class,"padded1-h")]/text()').extract())

        review['text'] = self.cleantxt(title + "\n" + body)

        review['movie_id'] = regex.search(r"wk\d+",response.url).group()
        review['review_id'] = regex.search(r"review\/(\d+)",response.url).groups()[0]

        return review

    def cleantxt(self,s):
        c = s.encode("utf-8").replace("<br />"," ")        
        c = c.replace('\r','')    
        c = c.replace('\t',' ')
        c = re.sub(r'\s+\n','\n',c)
        c = re.sub(r'\n\s+','\n',c)
        c = re.sub(r'\n+','\n',c)
        c = re.sub(r'\s+',' ',c)
        c = c.strip()       
        return c


