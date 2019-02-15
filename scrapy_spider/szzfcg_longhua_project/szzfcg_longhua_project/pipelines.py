# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import logging
logger = logging.getLogger(__name__)


class SzzfcgLonghuaProjectPipeline(object):
    """
    记得在setting处打开其权限，否则无法使用Pipeline
    """
    def process_item(self, item, spider):
        # for key in item.keys():
        #     print(key, ": ", item[key])
        logger.warning(item)  # 日志
        return item
