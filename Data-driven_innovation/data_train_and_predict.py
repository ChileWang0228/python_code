#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-30 1:43
@Author:ChileWang
@algorithm：
模型训练与预测
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import pickle
import os
import matplotlib.pyplot as plt
import random
data_to_train_and_test = 'data_to_train_and_test/'  # 训练集和测试集csv文件存放地址
image_to_save = 'image/'  # 图片存放文件夹
# pd.set_option('display.max_columns', None)  # 显示所有列


def one_hot_encode(encode_df_list, encode_list):
    """
    对标称型变量进行独热编码
    :param encode_df_list: 数据集
    :param encode_list: 需要独热编码的列
    :return:
    """
    feature = []
    for i in range(len(encode_df_list)):
        for col in encode_list:
            one_hot_columns = encode_df_list[i][col].unique()
            temp_df = pd.get_dummies(encode_df_list[i][col].replace('null', np.nan))  # 独热编码
            temp_df.columns = one_hot_columns
            encode_df_list[i][one_hot_columns] = temp_df  # 合并到原来的数据集
            if i == 1:
                feature.extend(one_hot_columns)  # 新生成的标称型特征
    return encode_df_list, feature


def apply_medium(row, pro_id, cou_id, ur_id, gender_id, temp_med):
    """
    中位数填充函数
    :param row:
    :return:
    """
    print(row)


def medium_fill(df_list, feature):
    """
    对数值型特征用中位数进行缺失值填充
    :param df_list:  训练集与验证集与测试集
    :param feature: 数值型特征
    :return:
    """

    for i in range(len(df_list)):
        for col in feature:
            # 中位数填充
            med = df_list[i][col].median()
            df_list[i][col].fillna(med, inplace=True)

            # temp = df_list[i]
            # temp_col_null = temp[temp[col].isnull()].copy()  # 取出该列的空值
            # temp_col_not_null = temp[temp[col].notnull()].copy()  # 取出该列的非空值
            # temp_col_null = temp_col_null[['provcd', 'countyid', 'urban', 'gender']].drop_duplicates()  # 去重
            #
            # province_id = list(temp_col_null['provcd'])  # 省份
            # county_id = list(temp_col_null['countyid'])  # 区县
            # urban_id = list(temp_col_null['urban'])  # 城镇与否
            # gender = list(temp_col_null['gender'])  # 性别
            # # 取该列非空集合中省份，区县，城镇，性别与该列为空值相似的的数据列，并取中位值填充空值
            # for j in range(len(province_id)):
            #     pro_id = province_id[j]
            #     cou_id = county_id[j]
            #     ur_id = urban_id[j]
            #     gender_id = gender[j]
            #     temp_median = float(temp_col_not_null[(temp_col_not_null['provcd'] == pro_id) &
            #                              (temp_col_not_null['countyid'] == cou_id) &
            #                              (temp_col_not_null['urban'] == ur_id) &
            #                              (temp_col_not_null['gender'] == gender_id)][[col]].median())
            #
            #     if np.isnan(temp_median):  # 若为空，则取该列的中位数填充
            #         temp_median = temp_col_not_null[col].median()
            #
            #     # 中位数填充
            #     df_list[i][col] = df_list[i].apply(apply_medium(row=df_list[i],pro_id=pro_id,cou_id=cou_id,ur_id=ur_id,gender_id=gender_id,temp_med=temp_median))
            #
            #     print(df_list[i][col].unique())

    return df_list


def var_delete(df_list, non_feature, num_feature):
    """
    方差删除法
    对标称型特征进行方差选择法，去掉取值变化小的特征
    :param df_list:
    :param non_feature: 适用于标称型特征
    :param num_feature: 适用于标称型特征
    :return:
    """
    var_del_model = VarianceThreshold(threshold=(.8 * (1 - .8)))
    var_del_model.fit_transform(df_list[0][non_feature])
    selected_index = var_del_model.get_support(indices=True)
    new_columns = []
    for index in selected_index:
        new_columns.append(non_feature[index])
    new_columns.extend(num_feature)  # 构造新的特征
    for i in range(len(df_list)):
        tem_df = df_list[i][new_columns].copy()  # 重新构造新的数据集
        if i < 1:
            tem_df['label'] = df_list[i]['label']
        df_list[i] = tem_df
    return df_list


def select_best_chi2(df_list):
    """
    利用卡方验证选择最佳的前ｋ个特征
    利用sklearn的库会让DataFrame变成二维列表
    :param df_list:
    :return:
    """
    columns = df_list[-1].columns.values.tolist()  # 测试集不包含label
    select_model = SelectKBest(chi2, k=25)  # 卡方验证选择模型
    X = MinMaxScaler().fit_transform(df_list[0][columns])  # 无量纲化
    select_model.fit_transform(X, df_list[0]['label'])
    selected_index = select_model.get_support(indices=True)  # 返回被选择特征的索引
    new_columns = []
    for index in selected_index:  # 重构新的特征变量
        new_columns.append(columns[index])
    for i in range(len(df_list)):
        tem_df = df_list[i][new_columns]
        if i < 1:
            tem_df['label'] = df_list[i]['label']
        df_list[i] = tem_df
    return df_list


def year_judge(row):
    """
    年份判断
    :param row:
    :return:
    """
    if row > -8:
        row = int(row)
    else:
        return row
    if (row > 0) and (row < 1950):
        return 4
    elif (row >= 1950) and (row < 1960):
        return 5
    elif (row >= 1960) and (row < 1970):
        return 6
    elif (row >= 1970) and (row < 1980):
        return 7
    elif (row >= 1980) and (row < 1990):
        return 8
    elif (row >= 1990) and (row < 2000):
        return 9
    elif (row >= 2000) and (row < 2100):
        return 10
    else:
        return 0


def year_partition(df_list, year_type):
    """
    对年份标称型特征进行划分：50后 60后　70后　80后　90后 00后
    :param df_list: 训练集与验证集与测试集
    :param year_type: 年份标称型特征
    :return:
    """
    for i in range(len(df_list)):
        for col in year_type:
            df_list[i][col] = df_list[i][col].apply(year_judge)
    return df_list


def pca_decompose(data_mat, n):
    """
    主成分分析降维
    :param data_mat: 原始矩阵
    :return:
    """
    pca = decomposition.PCA(n_components=n)
    pca_data_mat = pca.fit_transform(data_mat)
    reduction = pca.explained_variance_ratio_ .sum()  # 查看降维效果
    print(pca.explained_variance_ratio_)
    print(reduction)
    print(pca_data_mat.shape)
    return pca_data_mat


def check_model(x, y):
    """
    拟合的模型
    :return:返回最佳模型
    """
    gb_model = GradientBoostingClassifier(
                            learning_rate=0.05,  # 学习率
                            n_estimators=10000,  # 迭代次数
                            min_samples_leaf=70,
                            max_depth=7,
                            subsample=0.85,
                            random_state=10,
                            min_samples_split=100,
                            )
    lgb_model = lgb.LGBMClassifier(boosting_type="gbdt",
                                   num_leaves=30, reg_alpha=0, reg_lambda=0.,
                                   max_depth=-1, n_estimators=2500, objective='binary', metric='auc',
                                   subsample=0.9, colsample_bytree=0.7, subsample_freq=1,
                                   learning_rate=0.1,
                                   random_state=10)
    parm = {
            # 'n_estimators': range(5000, 6201, 200),  # 迭代范围
            # 'max_features': range(2, 5, 1), # 最大特征数
            # 'max_depth': range(3, 14, 2),
            # 'min_samples_split': range(100, 801, 200),
            # 'max_features': range(3, 5, 1),
            # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            }  # 参数
    g_search = GridSearchCV(estimator=gb_model,
                            param_grid=parm,  # 参数
                            scoring='roc_auc',
                            cv=10,  # 10折交叉验证  一定程度上用来减少过拟合
                            iid=False,  # 独立同分布
                            n_jobs=-1,  # -1 means using all processors
                            verbose=1  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
                            )
    g_search.fit(x, y)  # 运行网格搜索
    print(g_search.cv_results_['params'])  # 打印候选参数表
    print(g_search.cv_results_['mean_test_score'])  # 打印不同参数下的训练分数
    print(g_search.best_score_)  # 最佳分数
    print(g_search.best_params_)  # 最佳参数

    return g_search.best_estimator_  # 返回最佳模型


def save_model(model):
    """
    保存模型
    :param model:
    :return:
    """
    if not os.path.isfile('model/family_loss_model.pkl'):
        with open('model/family_loss_model.pkl', 'wb') as f:
            pickle.dump(model, f)


def get_model():
    """
    获得模型
    :return:
    """
    with open('model/family_loss_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def nominal_type_statistic(data_set, col):
    """
    标称型数据类型占比统计
    将标称型数据下的数据类型的占比进行统计，从大到小相加，当占比超过９５％的时候，剩余小于５％合并成一类，
    并返回合并形成一个集合，返回。
    :param data_set:
    :param col: 统计的列名
    :return:
    """
    temp = data_set[[col]].copy()
    temp['col_count'] = 1
    temp = temp.groupby([col], as_index=False).count()  # 基于col排序,将col_count统计
    data_num = data_set.shape[0]  # 总行数
    temp['col_rate'] = temp.col_count.apply(lambda x: x / data_num)
    temp.sort_values(by="col_rate", ascending=False, inplace=True)  # 对占比从大到小排序
    temp.reset_index(inplace=True)
    #  将对应的列和起百分比压缩成一个按值从大到小的字典
    col_list = list(temp[col])
    col_rate_list = list(temp['col_rate'])
    col_dict = dict(zip(col_list, col_rate_list))

    fina_data_list = list()  # 最终返回的数据类型列表
    threshold_rate = 0.85  # 阈值占比
    sum_rate = 0.0  # 累计占比
    name_rate = []  # 饼图的占比
    for key in col_dict.keys():
        if sum_rate >= threshold_rate:
            fina_data_list.append(key)  # 剩余的类别归并成一类
            name_rate.append(1 - sum_rate)
            break
        sum_rate += col_dict[key]
        name_rate.append(col_dict[key])
        fina_data_list.append(key)
    # 生成饼图
    # draw_pie_chart(fina_data_list, name_rate, col + ' pie chart')
    return fina_data_list


def produce_color(num):
    """
    随机生成num个不同的颜色
    :param num: 数量
    :return:
    """
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color_list = []
    while len(color_list) < num:
        color = ""
        for j in range(6):
            color += color_arr[random.randint(0, 14)]
        color = "#" + color
        if ~(color in color_list):  # 若该颜色不在该列表，则添加
            color_list.append(color)
    return color_list


def draw_bar_chart(name_list, heights, x_label, y_label, title):
    """
    绘制条形图
    :param name_list: 名字列表
    :param heights:　名字对应的数值列表
    :param x_label:　X轴名字
    :param y_label:　Y轴名字
    :param title:　表名
    :return:
    """
    plt.ion()  # 显示图片
    # 设置条形码的相应位置
    positions = [i for i in range(len(name_list))]
    plt.bar(positions, heights, color=produce_color(1), alpha=0.8, align='center', edgecolor='white')
    # 设置坐标轴名称
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(positions, name_list)  # 设置数据分类名称
    plt.title(title)  # 设置标题名称
    # 设置数字标签
    for x, y in zip(positions, heights):
        """
        其中，a, b+0.05表示在每一柱子对应x值、y值上方0.05处标注文字说明， 
        '%.0f' % b,代表标注的文字，即每个柱子对应的y值，
         ha='center', va= 'bottom'代表horizontalalignment（水平对齐）、
         verticalalignment（垂直对齐）的方式，fontsize则是文字大小。
        """
        plt.text(x, y+0.05, '%.0f' % y, ha='center', va='bottom')
    plt.legend([y_label], loc='upper right')  # 图例
    # plt.grid()  # 网格线
    plt.savefig(image_to_save + title)  # 保存图片
    plt.pause(1)  # 显示秒数
    plt.close()


def draw_pie_chart(name_list, name_rate, title):
    """
    生成饼图
    :param name_list: 名字列表
    :param name_rate: 名字对应的百分比
    :param title: 表名
    :return:
    """
    plt.ion()  # 显示图片
    color_list = produce_color(len(name_list))  # 生成互不相同的颜色
    explode_list = [0 for i in range(len(name_list))]  # 构造饼图分割表
    explode_list[-1] = 0.15  # 最后一个名字分割出来
    plt.pie(name_rate, labels=name_list, colors=color_list, explode=explode_list, startangle=60, autopct='%1.1f%%')
    plt.title(title)
    plt.legend(name_list, loc='upper left')
    plt.savefig(image_to_save + title)  # 保存图片
    plt.pause(1)  # 显示秒数
    plt.close()


def draw_roc_line(positive_rate, negative_rate, y_label_list, auc_scorce):
    """
    绘制roc折线图
    根据样本标签统计出
    正负样本的数量，假设正样本数量为P，负样本数量为N；接下来，把横轴的刻度
    间隔设置为1/N，纵轴的刻度间隔设置为1/P；再根据模型输出的预测概率对样本进
    行排序（从高到低）；依次遍历样本，同时从零点开始绘制ROC曲线，每遇到一
    个正样本就沿纵轴方向绘制一个刻度间隔的曲线，每遇到一个负样本就沿横轴方
    向绘制一个刻度间隔的曲线，直到遍历完所有样本，曲线最终停在（1,1）这个
    点，整个ROC曲线绘制完成。
    :param positive_rate: １/正样本个数
    :param negative_rate: １/负样本个数
    :param y_label_list: 基于预测概率从高到底排序的真实标签列表
    :param auc_scorce: auc面积
    :return:
    """
    plt.ion()  # 显示图片
    # 从(0, 0)开始
    x = 0
    y = 0
    x_list = [0]  # 横坐标
    y_list = [0]  # 纵坐标
    for label in y_label_list:
        if label == 1:  # 正样本
            y += positive_rate
            x_list.append(x)
            y_list.append(y)
        else:  # 负样本
            x += negative_rate
            x_list.append(x)
            y_list.append(y)
    # 到(1, 1)结束
    x_list.append(1)
    y_list.append(1)
    plt.plot(x_list, y_list, "b--", linewidth=1, label='auc score = ' + str(auc_scorce))  # (X轴，Y轴，蓝色虚线，线宽度, 图例)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()  # 让图例显效
    plt.savefig(image_to_save + 'roc_curve')  # 保存图片
    plt.pause(5)  # 显示秒数
    plt.close()


def process_data():
    """
    对数据集进行标准化处理
    :return:
    """
    train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
    test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')

    # 生成训练集和测试集的数量条形图
    name_list = ['train set', 'test set']
    heights = [train_set.shape[0], test_set.shape[0]]
    x_label = 'data set'
    y_label = 'amount of data set'
    title = 'amount of train set and test set'
    # draw_bar_chart(name_list, heights, x_label, y_label, title)

    numerical_type = ['qa1age', 'qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc',
                      'land_asset', 'total_asset', 'expense', 'familysize', 'fproperty']  # 数值型特征
    nominal_type = ['provcd', 'countyid', 'urban', 'gender', 'qd3', 'qe1_best', 'qe507y',
                    'qe211y', 'qp3', 'wm103', 'wn2', 'birthy_best', 'alive_a_p', 'tb3_a_p',
                    'tb4_a_p', 'alive_a_f', 'alive_a_m', 'tb6_a_f', 'tb6_a_m']  # 标称型特征

    # 生成数值特征和标称特征的饼图和条形图
    name_list = ['nominal_type', 'numerical_type']
    name_rate = [len(nominal_type), len(numerical_type)]
    title = 'numerical_type and nominal_type'
    # draw_pie_chart(name_list, name_rate, title)
    x_label = 'type of features'
    y_label = 'amount of features'
    # draw_bar_chart(name_list, name_rate, x_label, y_label, title)

    year_type = ['qe507y', 'qe211y', 'birthy_best']  # 年份型标称变量, 对其进行分桶
    df_list = [train_set, test_set]  # 将训练集和测试集放入列表中
    df_list = year_partition(df_list, year_type)  # 将年份的分桶
    # 标称型缺失值统一为-30
    for df in df_list:
        df[nominal_type] = df[nominal_type].fillna(-30)
    family_loss_set = df_list[0][df_list[0]['label'] == 1].copy()  # 抽出label为１的个人流失数据

    # 生成训练集label=1和label=0的饼图和条形图
    name_list = ['0', '1']
    name_rate = [df_list[0][df_list[0]['label'] == 0].shape[0], df_list[0][df_list[0]['label'] == 1].shape[0]]
    title = 'percentage of label type in train set'
    # draw_pie_chart(name_list, name_rate, title)
    title = 'amount of label type in train set'
    x_label = 'type of labels'
    y_label = 'amount of labels'
    # draw_bar_chart(name_list, name_rate, x_label, y_label, title)

    for nt in nominal_type:  # 对特定的标称型特征进行占比统计
        fina_key_list = nominal_type_statistic(family_loss_set, nt)  # 得到最终的占比统计类别与剩余类别　
        # 对训练集和测试集合相应的特征进行新的分类, 若类在于fina_key_list当中，则继续应用，否则将以得到的剩余类别将其分类
        df_list[0][nt] = df_list[0][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])
        df_list[1][nt] = df_list[1][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])

    # 数值型特征缺失值补充
    df_list = medium_fill(df_list, numerical_type)

    # 特征选择
    df_list = var_delete(df_list, nominal_type, numerical_type)  # 方差选择法
    df_list = select_best_chi2(df_list)  # 卡方验证选择法

    # 标称型独热编码
    columns = df_list[1].columns.values.tolist()  # 测试集不包含label
    new_nominal_type = list(set(columns) & set(nominal_type))  # 新的标称型列
    new_numerical_type = list(set(columns) & set(numerical_type))  # 新的数值型列
    df_list, new_nominal_type = one_hot_encode(df_list, new_nominal_type)  # 独热编码
    for df in df_list:
        print(df.head(5))
    new_feature = new_nominal_type + new_numerical_type  # 新的特征列
    print(new_feature)

    # PCA降维
    n_components = 5  # pca的维度
    new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
    pca_train_set = pd.DataFrame(pca_decompose(df_list[0][new_feature].values, n_components), columns=new_pridictors)  # DF转矩阵
    pca_test_set = pd.DataFrame(pca_decompose(df_list[1][new_feature].values, n_components), columns=new_pridictors)  # DF转矩阵
    # 将相应的id补上
    pca_train_set['label'] = df_list[0]['label']
    pca_train_set['pid'] = train_set['pid']
    pca_train_set['fid'] = train_set['fid']

    pca_test_set['pid'] = test_set['pid']
    pca_test_set['fid'] = test_set['fid']

    # 建模
    new_train_set, new_valid_set = train_test_split(pca_train_set, test_size=0.2, stratify=pca_train_set['label'], random_state=100)
    new_test_set = pca_test_set

    # 训练
    x = new_train_set[new_pridictors]
    y = new_train_set['label']
    model = check_model(x, y)
    # model = get_model()

    # 验证
    x_valid = new_valid_set[new_pridictors]
    y_valid = new_valid_set['label']
    y_pred = model.predict(x_valid)
    y_pred_prob = model.predict_proba(x_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred_prob)
    print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
    print('AUC Score (Train): %.4g' % auc_score)

    # 绘制roc曲线
    positive_rate = 1/(new_valid_set[new_valid_set['label'] == 1].shape[0])  # 正样本刻度
    negative_rate = 1/(new_valid_set[new_valid_set['label'] == 0].shape[0])  # 负样本刻度
    temp_dict = {'label': y_valid, 'pred_prob': y_pred_prob}
    temp = pd.DataFrame(temp_dict)
    temp.sort_values('pred_prob', inplace=True, ascending=False)  # 基于预测概率倒序排序
    draw_roc_line(positive_rate, negative_rate, list(temp['label']), auc_score)

    # 测试
    test_x = new_test_set[new_pridictors]
    test_y_pred_prob = model.predict_proba(test_x)[:, 1]
    test_y_label = model.predict(test_x)
    new_test_set['pred'] = test_y_pred_prob
    new_test_set['label'] = test_y_label
    summit_file = new_test_set[['pid', 'fid', 'pred', 'label']].copy()
    summit_file.columns = ['pid', 'fid', 'pred', 'label']
    summit_file.to_csv('summit.csv', columns=['pid', 'fid', 'pred', 'label'], index=False)  # 不要索引, header=False 不要列头

    # 保存模型
    save_model(model)


def get_fina_result():
    """
    生成最终流失的1500个fid
    :return:
    """
    # get_model()
    result = pd.read_csv('summit.csv')
    result = result[['fid', 'pred']].copy()
    result['count'] = 1
    temp = result.groupby(['fid'], as_index=False).count()  # 统计每个fid出现的次数
    res = (result.groupby(['fid'], as_index=False)['pred'].sum()).copy()  # 基于fid求和流失概率
    res['fid_count'] = temp['count']
    res['final_probability'] = res.apply(lambda x: x['pred'] / x['fid_count'], axis=1)  # 二者相除得出流失的概率
    res = (res.sort_values('final_probability', ascending=False)).copy()  # 降序排序
    res.to_csv('final_disappear_fid.csv', columns=['fid', 'final_probability'], index=False)


if __name__ == '__main__':
    process_data()
    # get_fina_result()


