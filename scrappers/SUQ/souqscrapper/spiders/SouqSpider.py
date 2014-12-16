#encoding:utf-8 
'''
Created on 20.8.2014
python 2.7.3
@author: hadyelsahar 
'''
import scrapy

from souqscrapper.items  import * 

class SouqSpider(scrapy.Spider):   
    
    name = "souqspider"
    start_urls = []
    allowed_domains = ["souq.com"]

    categories = ["accessories","air-treatment","analogue-camera","antique","athletic-wear","audio-accessories","baby-accessories","baby-bag","baby-bath-skin-care","baby-clothes","baby-gear","baby-gift-set","baby-safety-health","baby-toy-accessories","backpacks","Bag-carrying-Case","barbecue-tool-grill-accessories","barcode-reader","bath-body","bathroom-equipment","beauty-gift-set","beauty-tools-accessories","bedding","binoculars","books","books-manuscript","boots","bracelets","business-bags","cabinet","cables","camcorder","camera-camcorder-accessories","camping-goods","car-audio","car-audio-video-accessories","car-care-product","card-reader-writer","car-navigation","car-video","cd-player-recorder","cd-recording-media","chair-bench","clock-compasses","clock-radio","coins","comic-graphic-novel","compters-fan","computer","computer-casing","computer-monitor","computer-mouse","computer-peripheral","cooking-set","cooling-pad","cpu-ram","deep-fryer","dental-care","diapers","digital-camera","digital-fever-thermometer","digital-photo-frame","docking-station","drawing-painting","dresses","dvd-blu-ray-player","dvd-vcr-combos","earrings","ebook-reader","educational-book","electrical-personal-machine","electric-slicer","electronic-flash","eyewear","fans","feeding-diapering-bathing","fiction-literature","flat-panel-display","food-preparation","food-supplement","games","games-console","games-console-accessories","garden-decoration","garden-equipment-watering","garden-light","gps-accessories","gps-navigator","gps-receiver","hair-care","hair_electronics","handbags","handcraft-sculpture-carving","hand-tool","hard-disk","headphone","headset","health-personal-care","heater","home-decor","home-supplies","hot-beverage-maker","ink-cartridges","interchangeable-lense","ironing-accessories","irons","islamic-ethnic-digital-art","jacket-coats","jewelry-accessories","jewelry-set","juice-extractor","kettle","keyboard","keys-key-chains","kids-book","kitchen-dining","kitchen-scale","lamp","laptop-charger","laptop-notebook","laptop-usb-memory","lcd-led-dlp-tv","loose-gemstones-diamond","loud-speaker","luggage","makeup","map-atlas-globe","maternity-wear","media-gateway","memory-card","men-grooming","men-jewleries","messenger-bags","microphone","mobile-phone","mobile-phone-accessories","mobile-usb-memory","motherboard","movies-plays-series","mp3-mp4-player-accessories","mp3-player","multifunction-devices","musical-instrument","music-cd","natural-nutrition-products","necklace-pendant","netbook","network-card-adapter","networking-tool","network-switch","office-equipment","office-supplies","optical-drive","pants","pc-media-player-speaker","perfumes-fragrances","personal-scale","pet-supplies","photograph","plasma-tv","portable-audio-player","portable-radio","portable-tv","poster","power-supply","power-tool","printer","printer-scanner-accessories","prints","projector","projector-accessories","receiver-amplifier","rechargeable-batteries","recording-studio-equipment","rice-cooker","rings","router","rugs-carpets","sandals","sandwich-waffle-makers-grill","satellite-receiver","scanner","school-bags","security-surveillance-system","server","shoes","skin-care","skirts","sleepwears","slippers","small-appliance","small-medical-equipment","smoking-accessories","software","sound-card","sporting-goods","sport-nutrition","stamps","stationary","steam-cleaner","stereo-system-equalizer","still-film","swimwears","tablet","tablet-accessories","telephone","telephone-accessories","toaster","tops","toys","tuner","tv-mounts","tv-satellite-accessories","underwears","uniforms","uninterruptible-power-supply-ups","universal-remote-control","usb-drive-memory","vacuum-cleaner","vacuums-floor-care","vcd-vcp-vcr-player","video-accessories","video-card","video-tape","vitamin-mineral","watch-accessories","watches","webcam","wigs","women-lingerie"]
    
    for w in [('saudi','sa-ar'),('uae','ae-ar'),('egypt','eg-ar')]:
        for c in categories:
            start_urls.append('http://'+w[0]+'.souq.com/'+w[1]+'/'+c+'/l/')

    def parse(self,response):
       
        for sel in response.css('#ItemResultList .single-item-browse'):
            
            psel = sel.css('table>tr>td>a')                      
            url = psel.xpath('@href').extract()[0]

            yield scrapy.Request(url, callback=self.parse_review)
        
        nexturl = response.xpath('//*[@id="box-results"]/div/div[4]\
        	/div/div/div/div[1]/div/div/ul/li[11]/a/@href').extract()

        if len(nexturl) > 0 :
        	yield scrapy.Request(nexturl[0], callback=self.parse)


    def parse_review(self,response):        

        category = response.xpath(u'//span[contains(@itemprop,"title")]/text()').extract()

        productId =  response.xpath(u'//table/tr/td[contains(text(),\
        	"الرقم المميز للسلعة")]/../td[2]/text()').extract()[0]

        for sel in response.xpath(u'//div[contains(@class,"review-page")]'):

            item = Review()
            item['productId'] = productId
            item['pageLink'] = response.url
            item['country'] = response.url.replace("http://","").split(".")[0]
            
            title  = sel.css('.overhidden .fl.txt13').xpath('text()').extract()
            if len(title) > 0 :
                item['title'] = title[0]

            item['rating'] = len(sel.css('.overhidden>ul>li>.color-rating'))
            item['category'] = category
            item['isowner'] =  1 if len(sel.xpath(u'//div/span/strong[contains(text(),\
                "المستخدم قام بشراء هذه السلعة")]')) > 0 else 0 
            usrid = sel.css('.overhidden.marb-15>a').xpath('text()').extract()[0]
            item['userid'] =  usrid.strip()

            postext = sel.css('.marb-10.txt13').xpath(u'//em[contains(text(),\
            	"ما إيجابيات هذه السلعة:")]/..').css('.padt-5').xpath('text()').extract()
            if len(postext) > 0:
                item['posText'] = postext[0].strip()
            negtext = sel.css('.marb-10.txt13').xpath(u'//em[contains(text(),\
            	"ما سلبيات هذه السلعة:")]/..').css('.padt-5').xpath('text()').extract()
            if len(negtext) > 0 :
                item['negText'] = negtext[0].strip()

            othertext = sel.css('.marb-10.txt13').xpath(u'//em[contains(text(),\
            	"التقييم:")]/..').css('.padt-5').xpath('text()').extract()
            if len(othertext) > 0 :
                item['otherText'] = othertext[0].strip()
            
            item['recommend'] = 1 if len(sel.xpath(u'//div[contains(text(),\
            	"انصح بهذه السلعة لصديق")]/..').extract()) > 0 else 0 
            rationandvote = sel.css('#ratio_and_total').xpath('strong/text()').extract()
            item['voters'] = rationandvote[1]
            item['usefulness'] = rationandvote[0]
    
            yield item


