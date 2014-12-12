# -*- coding: utf-8 -*-
import scrapy

class Review(scrapy.Item): 
    movie_id = scrapy.Field()
    user_id = scrapy.Field()
    review_id = scrapy.Field()
    text = scrapy.Field()
    score = scrapy.Field()