#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-3-18 9:50
@Author:ChileWang
@algorithm：
利用附件 1 所给数据，提取并分析车辆的运输路线以及其在运输过程中的速度、加
速度等行车状态。提交附表中 10 辆车每辆车每条线路在经纬度坐标系下的运输线路图及对
应的行车里程、平均行车速度、急加速急减速情况。
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
vehicles_data_dir = '/home/chilewang/Desktop/example_data_100vehicles/'  # 存放车辆信息的文件夹
image_to_save = 'image/'


def draw_driving_path(x_list, y_list, legend, image_name, x_label='x', y_label='y'):
    """
    绘制折线图
    :param x_list: 横坐标列表
    :param y_list: 纵坐标列表
    :param legend: 图例名字
    :param image_name: 图片名字
    :param x_label: 横坐标的名字
    :param y_label: 纵坐标的名字
    :return:
    """
    plt.ion()  # 显示图片
    plt.plot(x_list, y_list, "b", linewidth=1, label=legend)  # (X轴，Y轴，蓝色虚线，线宽度, 图例)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(image_name)
    plt.legend(loc='best')  # 让图例显效
    plt.pause(5)  # 显示秒数
    plt.savefig(image_to_save + image_name)  # 保存图片
    plt.close()


if __name__ == '__main__':
    files_list = os.listdir(vehicles_data_dir)  # 该文件夹下的所有文件名
    vehicles_data = pd.read_csv(vehicles_data_dir + files_list[0])
    vehicles_data['location_date'] = vehicles_data['location_time'].apply(lambda x: x.split(' ')[0])
    location_date = vehicles_data['location_date'].unique()  # 行车日期
    vehicle_plate_number = vehicles_data['vehicleplatenumber'][0]  # 车牌号
    for date in location_date:
        lng = vehicles_data[vehicles_data['location_date'] == date]['lng']
        lat = vehicles_data[vehicles_data['location_date'] == date]['lat']
        image_name = vehicle_plate_number + ' ' + date + ' driving path record'  # 图片名字
        draw_driving_path(list(lng), list(lat), vehicle_plate_number, image_name, x_label='lng', y_label='lat')



