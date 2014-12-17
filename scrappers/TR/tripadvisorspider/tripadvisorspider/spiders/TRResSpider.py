#encoding:utf-8 
'''
Created on 20.8.2014
python 2.7.3
@author: hadyelsahar 
'''
import scrapy
from scrapy.shell import inspect_response

from tripadvisorspider.items  import * 


class TR_ResSpider(scrapy.Spider):    
    name = "tripadvisorrestaurantspider"
    start_urls = ['http://www.tripadvisor.com.eg/AllLocations-g1-c1-Hotels-World.html']                  
    # start_urls = ['http://www.tripadvisor.com.eg/AllLocations-g294201-c1-Hotels-Cairo_Cairo_Governorate.html'] #test egypt hotels 

    baseurl = 'http://www.tripadvisor.com.eg'
    allowed_domains = ["tripadvisor.com.eg"]

    def parse(self,response):        
        
        for location in response.xpath("//a[contains(@href,'/AllLocations')]/@href").extract():
            yield scrapy.Request(self.baseurl+location, callback=self.parse)

        for location in response.xpath("//a[contains(@href,'/Restaurants-')]/@href").extract():
            yield scrapy.Request(self.baseurl+location, callback=self.parse)

        for i in response.xpath("//div[@id='BODYCON']//a[contains(@href,'/Restaurant_Review-')]/@href").extract():            
            yield scrapy.Request(self.baseurl+i, callback=self.parse_restaurant)


    def parse_restaurant(self,response):
        
        hotel_name = response.css('#HEADING').xpath('./text()').extract()[0]

        for entry in response.css('.reviewSelector'):
            # non translated reviews are got in default html 
            #(no need to parse js or call Api, luckily !!)
            # translated reviews are with empty .reviewSelector
            title = entry.css('.quote').xpath('./a')
            if len(title) > 0 :
                r = Review()
                r['hotel_name']= hotel_name
                r['hotel_url'] = response.url

                user = entry.css(".member_info").xpath("./div/@id")
                if len(user) > 0 :
                    r['userid'] = user.extract()[0].split("-")[0]

                r['review_id'] = entry.xpath('./@id').extract()[0] 
                r['title'] = title.xpath('./span/text()').extract()[0]
                r['review_url'] = title.xpath('./@href').extract()[0]                
                scoretxt = entry.css('.rating').xpath('./span/img/@alt').extract()[0]
                r['score'] = int(scoretxt.split(" ")[0].strip())
                
                #filling review text if it has more send to parse_more
                if len(entry.css('.entry .partnerRvw')) > 0 :                                    
                    yield scrapy.Request(self.baseurl+r['review_url'], callback=self.parse_more_review, meta={"item":r})
                else :                     
                    r['text'] = entry.css('.entry').xpath('./p/text()').extract()[0]
                    yield  r

        # parse next page
        # if number of titles < 10 no need to parse next page 
        # because all reviews will not be in Arabic Language
        rev_ar = response.css('.reviewSelector .quote').xpath('./a/span/text()').extract()        
        if len(rev_ar) == 10 :
            nextlink = response.css('.sprite-pageNext').xpath('./@href').extract()
            if len(nextlink) > 0 :
                yield scrapy.Request(self.baseurl+nextlink[0], callback=self.parse_restaurant)

    def parse_more_review(self,response):
        r = response.meta["item"]
        r['text']=" ".join(response.xpath(".//p[@id = \"%s\" ]/text()"%r['review_id']).extract())

        yield r 



