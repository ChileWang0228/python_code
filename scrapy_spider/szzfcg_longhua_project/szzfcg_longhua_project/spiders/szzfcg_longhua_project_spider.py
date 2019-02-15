# -*- coding: utf-8 -*-
import scrapy
from szzfcg_longhua_project.items import SzzfcgLonghuaProjectItem


class SzzcfgLonghuaProjectSpider(scrapy.Spider):
    name = "szzcfg_longhua_project"  # 爬虫名称
    allowed_domains = ["szzfcg.cn"]  # 允许域名
    start_urls = [
        "http://www.szzfcg.cn/portal/documentView.do?method=view&id=282377090"
    ]

    def parse(self, response):
        website_name = '深圳市龙华区公共资源交易中心'  # 网站名称
        publish_time = ''  # 发布时间
        ip = ''  # 爬虫ip
        html_content = response.xpath(".//*[@id='contentDiv']//*/tbody//*/tbody/tr/td")  # 要抽取的html所有内容

        bidding_name_position = html_content.xpath("./h3/span")[1]
        bidding_name = bidding_name_position.xpath('string(.)').extract()[0]  # 项目名称

        bidding_url_position = html_content.xpath("./a")[-1]
        bidding_url = bidding_url_position.xpath("./@href").extract()[0]  # 项目链接

        info_list = html_content.xpath('string(.)').extract()  # 所有的公告内容
        bidding_content = info_list[0] + info_list[-1]  # 公告内容

        bidding_tenderee_position = html_content.xpath("./ol")[-1].xpath('./li')[-1]
        bidding_tenderee = bidding_tenderee_position.xpath('string(.)').extract()[0]  # 采购人

        bidding_block = html_content.xpath("./ol")[0].xpath('./li').xpath('string(.)').extract()[0]  # 项目类别
        bidding_open_time = html_content.xpath("./ol")[0].xpath('./li').xpath('string(.)').extract()[-2]  # 开标时间
        bidding_end_time = html_content.xpath("./ol")[0].xpath('./li').xpath('string(.)').extract()[-1]  # 投标截止时间
        bidding_agency = ""  # 代理机构
        item = SzzfcgLonghuaProjectItem(website_name=website_name, publish_time=publish_time, ip=ip,
                                          bidding_name=bidding_name, bidding_url=bidding_url,
                                          bidding_content=bidding_content, bidding_tenderee=bidding_tenderee,
                                          bidding_block=bidding_block, bidding_open_time=bidding_open_time,
                                          bidding_end_time=bidding_end_time, bidding_agency=bidding_agency
                                          )
        yield item


