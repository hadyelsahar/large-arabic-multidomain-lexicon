# -*- coding: utf-8 -*-
import scrapy

class Review(scrapy.Item):    
    title = scrapy.Field()
    hotel_name = scrapy.Field()
    hotel_url = scrapy.Field()
    review_url = scrapy.Field()
    text = scrapy.Field()    
    score = scrapy.Field()
    review_id = scrapy.Field()
