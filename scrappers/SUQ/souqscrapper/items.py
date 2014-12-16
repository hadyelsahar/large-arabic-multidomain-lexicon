# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class Review(scrapy.Item):

    productId = scrapy.Field()   
    country = scrapy.Field()
    pageLink = scrapy.Field()
    category = scrapy.Field()
    title = scrapy.Field()
    rating = scrapy.Field()
    posText = scrapy.Field()
    negText = scrapy.Field()
    otherText = scrapy.Field()
    userid = scrapy.Field()
    voters = scrapy.Field()
    usefulness = scrapy.Field()
    isowner = scrapy.Field()
    recommend = scrapy.Field()