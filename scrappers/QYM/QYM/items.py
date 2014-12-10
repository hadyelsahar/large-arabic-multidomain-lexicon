# -*- coding: utf-8 -*-
import scrapy

class Review(scrapy.Item):    
    restaurant_id = scrapy.Field()
    user_id = scrapy.Field()
    text = scrapy.Field()
    polarity = scrapy.Field()
   
