# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
"""
基础信息:
    站点名称：以website.xlsx文件中的网站名称为准
    爬虫ip:爬虫机器的ip
招标信息:
    项目名称：项目名称，非公告名称
    公告内容：抓取以HTML展示的公告内容的文本
    项目类别：部分站点有对项目进行分类：设备采购/日常维修/其他
抽取以下信息，若无则留空
"""


class SzzfcgLonghuaProjectItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    website_name = scrapy.Field()  # 网站名称
    publish_time = scrapy.Field()  # 发布时间
    ip = scrapy.Field()  # 爬虫ip

    bidding_name = scrapy.Field()  # 项目名称
    bidding_url = scrapy.Field()  # 项目链接
    bidding_content = scrapy.Field()  # 公告内容
    bidding_block = scrapy.Field()  # 项目类别
    bidding_tenderee = scrapy.Field()  # 采购人
    bidding_agency = scrapy.Field()  # 代理机构
    bidding_open_time = scrapy.Field()  # 开标时间
    bidding_end_time = scrapy.Field()  # 投标截止时间
    bidding_purchase_mode = scrapy.Field()  # 采购方式
