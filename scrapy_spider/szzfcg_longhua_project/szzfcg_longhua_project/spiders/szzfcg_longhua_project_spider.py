# -*- coding: utf-8 -*-
import scrapy
import time
from szzfcg_longhua_project.items import SzzfcgLonghuaProjectItem

timeStamp = int(time.time())
print(timeStamp)


class SzzcfgLonghuaProjectSpider(scrapy.Spider):
    name = "szzcfg_longhua_project"  # 爬虫名称
    allowed_domains = ["szzfcg.cn"]  # 允许域名
    start_urls = [
        "http://www.szzfcg.cn/portal/topicView.do?method=viewList&id=500100201&siteId=11&tstmp=15%3A40%3A03%20GMT%2B0800%20(Hong%20Kong%20Standard%20Time)"
    ]

    def parse(self, response):
        project_list_position = response.xpath(".//*/ul/li")  # 采购信息列表位置
        for project in project_list_position:
            project_list_url = project.xpath("./a/@href")  # 获取具体的招标项目链接
            website_name = project.xpath("./a/@title")  # 网站名称
            item = SzzfcgLonghuaProjectItem(website_name=website_name)
            request = scrapy.Request(url=project_list_url, callback=self.parse_detail)
            request.meta['item'] = item  # 将item暂存
            yield request

        page_num_position = response.xpath(".//*[@class='page']//*/a")  # 页码位置
        total_page_num = page_num_position[6].xpath(".//*/b/text()").extract()[0]   # 总页码
        print(total_page_num)
        new_page_num = page_num_position[7].xpath("./@onclick").extract()[0]  # 下一页页码
        # 提取下一页页码的数字
        new_page_num = new_page_num.split('(')[1]
        new_page_num = new_page_num.split(',')[0]
        print(new_page_num)

        # 构建下一页的链接
        if int(new_page_num) < (int(total_page_num) + 1):   # 总页码 + 1页 用来判断是否到末页
            new_page_url = "http://www.szzfcg.cn/portal/topicView.do?method=viewList&id=500100201" \
                           "&page=" + str(new_page_num) + \
                           "&siteId=11&" + \
                           "tstmp=" + str(timeStamp) + "GMT%2B0800%20(Hong%20Kong%20Standard%20Time)"
            print(new_page_url)
            yield scrapy.Request(url=new_page_url, callback=self.parse)

    def parse_detail(self, response):
        """
        处理具体的招标项目信息
        :return:
        """
        # website_name = '深圳市龙华区公共资源交易中心'  # 网站名称
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

        # 重新构造item
        item = response.meta['item']
        website_name = item['website_name']
        item = SzzfcgLonghuaProjectItem(website_name=website_name, publish_time=publish_time, ip=ip,
                                          bidding_name=bidding_name, bidding_url=bidding_url,
                                          bidding_content=bidding_content, bidding_tenderee=bidding_tenderee,
                                          bidding_block=bidding_block, bidding_open_time=bidding_open_time,
                                          bidding_end_time=bidding_end_time, bidding_agency=bidding_agency
                                          )
        yield item


