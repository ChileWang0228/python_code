#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-19 15:21
@Author:ChileWang
@algorithm：
微信聊天机器人
"""
from wxpy import *
import requests
import json


def auto_reply(text):
    """
    图灵机器人自动回复
    :param text:
    :return:
    """
    url = "http://www.tuling123.com/openapi/api"
    api_key = '24bbf8dcb1194b249df0ce4705caea5e'
    payload = {
        'key': api_key,
        'info': text,
        'userid': '381514'
    }
    r = requests.post(url, data=json.dumps(payload))
    result = json.loads(r.content)
    return '[Robot Made By Chile] ' + result['text']


bot = Bot(console_qr=True, cache_path=True)


@bot.register(bot.self, except_self=False)
def forward_myself(msg):
    """
    回复自己
    :param msg:
    :return:
    """
    return auto_reply(msg.text)


@bot.register(chats=[Friend])
def reply_friends(msg):
    """
    回复朋友
    :param msg: 接收到的信息
    :return:
    """
    print('[Received]:' + str(msg))
    if msg.type != 'Text':
        ret = '[坏笑][坏笑]'
    else:
        ret = auto_reply(msg.text)
    print('[Send]:' + str(ret))
    return ret


@bot.register(Group, TEXT)
def reply_group_at(msg):
    """
    回复群组＠我的人
    :param msg:
    :return:
    """
    print('[Received]:' + str(msg))
    if msg.is_at:
        if msg.type != 'Text':
            ret = '[坏笑][坏笑]'
        else:
            ret = auto_reply(msg.text)
        print('[Send]:' + str(ret))
        return ret


dormitory = bot.groups().search('410')[0]  # 研究生宿舍群


@bot.register(dormitory)
def reply_group(msg):
    """
    回复群组＠我的人
    :param msg:
    :return:
    """
    print('[Received]:' + str(msg))
    if msg.type != 'Text':
        ret = '[坏笑][坏笑]'
    else:
        ret = auto_reply(msg.text)
    print('[Send]:' + str(ret))
    return ret


embed()  # 进入命令行,让程序一直执行






