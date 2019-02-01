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
data_to_train_and_test = 'data_to_train_and_test/'  # 训练集和测试集csv文件存放地址
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
            # 独热编码后的列,取三者的并集是避免三个训练集中的某些情况模型不包含
            one_hot_columns = encode_df_list[i][col].unique()
            temp_df = pd.get_dummies(encode_df_list[i][col].replace('null', np.nan))  # 独热编码
            temp_df.columns = one_hot_columns
            encode_df_list[i][one_hot_columns] = temp_df  # 合并到原来的数据集
            if i == 0:
                feature.extend(one_hot_columns)  # 新生成的标称型特征
    return encode_df_list, feature


def medium_fill(df_list, feature):
    """
    对数值型特征用中位数进行缺失值填充
    :param df_list:  训练集与验证集与测试集
    :param feature: 数值型特征
    :return:
    """
    for i in range(len(df_list)):
        for col in feature:
            med = df_list[i][col].median()
            df_list[i][col] = df_list[i][col].fillna(med)
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
        if i < 2:
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
    columns = df_list[2].columns.values.tolist()  # 测试集不包含label
    select_model = SelectKBest(chi2, k=25)  # 卡方验证选择模型
    X = MinMaxScaler().fit_transform(df_list[0][columns])  # 无量纲化
    select_model.fit_transform(X, df_list[0]['label'])
    selected_index = select_model.get_support(indices=True)  # 返回被选择特征的索引
    new_columns = []
    for index in selected_index:  # 重构新的特征变量
        new_columns.append(columns[index])
    for i in range(len(df_list)):
        tem_df = df_list[i][new_columns]
        if i < 2:
            tem_df['label'] = df_list[i]['label']
        df_list[i] = tem_df
    return df_list


def year_judge(row):
    """
    年份判断
    :param row:
    :return:
    """
    if row >= -9:
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
            if i != 0 and col == year_type[0]:
                df_list[i][col] = df_list[i][col].apply(lambda x: 0 if x > 6 else x)
            if i != 0 and col == year_type[1]:
                df_list[i][col] = df_list[i][col].apply(lambda x: 0 if x in [4, 7, 10] else x)
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
    print(reduction)
    print(pca_data_mat.shape)
    return pca_data_mat


def check_model(x, y):
    """
    拟合的模型
    :return:返回最佳模型
    """
    gb_model = GradientBoostingClassifier(
                            learning_rate=0.1,  # 选择模型
                            n_estimators=2500,  # 迭代次数
                            min_samples_leaf=70,
                            max_depth=7,
                            subsample=0.85,
                            random_state=10,
                            min_samples_split=1000
                            )
    lgb_model = lgb.LGBMClassifier(boosting_type="gbdt",
                                   num_leaves=30, reg_alpha=0, reg_lambda=0.,
                                   max_depth=-1, n_estimators=2500, objective='binary', metric='auc',
                                   subsample=0.9, colsample_bytree=0.7, subsample_freq=1,
                                   learning_rate=0.1,
                                   random_state=2018)
    parm = {
            # 'max_features': range(7, 20, 2), # 最大特征数
            }  # 参数
    g_search = GridSearchCV(estimator=lgb_model,
                            param_grid=parm,  # 参数
                            cv=5,  # 五折交叉验证
                            n_jobs=-1,  # -1 means using all processors
                            verbose=1  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
                            )
    g_search.fit(x, y)
    print(g_search.best_score_)
    print(g_search.best_params_)

    return g_search.best_estimator_  # 返回最佳模型


def save_model(model):
    """
    保存模型
    :param model:
    :return:
    """
    if not os.path.isfile('bank_loan_modle.pkl'):
        with open('modle_1.pkl', 'wb') as f:
            pickle.dump(model, f)


def get_model():
    """
    获得模型
    :return:
    """
    with open('modle_1.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def process_data():
    """
    对数据集进行标准化处理
    :return:
    """
    train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
    valid_set = pd.read_csv(data_to_train_and_test + 'valid_set.csv')
    test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')
    numerical_type = ['qa1age', 'qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc',
                      'land_asset', 'total_asset', 'expense', 'familysize', 'fproperty']  # 数值型特征
    nominal_type = ['provcd', 'countyid', 'urban', 'gender', 'qd3', 'qe1_best', 'qe507y',
                    'qe211y', 'qp3', 'wm103', 'wn2', 'birthy_best', 'alive_a_p', 'tb3_a_p',
                    'tb4_a_p', 'alive_a_f', 'alive_a_m', 'tb6_a_f', 'tb6_a_m']  # 标称型特征
    year_type = ['qe507y', 'qe211y', 'birthy_best']  # 年份型标称变量, 对其进行分桶
    df_list = [train_set, valid_set, test_set]

    df_list = year_partition(df_list, year_type)  # 将年份的分桶

    # 对省份特征统一命名
    common = list(df_list[0]['provcd'].unique())
    common.remove(-9)  # 将-9删除，当出现不在列表里的元素时,统一将其归类为0
    for i in range(len(df_list)):
        df_list[i]['provcd'] = df_list[i]['provcd'].apply(lambda x: x if x in common else 0)

    # 对县区id进行统一命名
    common = list(df_list[0]['countyid'].unique())
    common.remove(-9)  # 将-9删除，当出现不在列表里的元素时,统一将其归类为0
    for i in range(len(df_list)):
        df_list[i]['countyid'] = df_list[i]['countyid'].apply(lambda x: x if x in common else -20)

    # 对gender, qd3, qp3进行统一命名
    common = list(df_list[0]['gender'].unique())
    common.remove(0)  # 将0删除，当出现不在列表里的元素时,统一将其归类为-20
    for i in range(len(df_list)):
        df_list[i]['gender'] = df_list[i]['gender'].apply(lambda x: x if x in common else -20)

    common = list(df_list[0]['qd3'].unique())
    common.remove(0)  # 将0删除，当出现不在列表里的元素时,统一将其归类为-20
    for i in range(len(df_list)):
        df_list[i]['qd3'] = df_list[i]['qd3'].apply(lambda x: x if x in common else -20)
    # 同理
    process_type = ['qp3', 'wm103', 'wn2', 'tb4_a_p', 'alive_a_f', 'alive_a_m', 'tb6_a_f', 'tb6_a_m']
    for col in process_type:
        common = list(df_list[0][col].unique())
        common.remove(-8)
        for i in range(len(df_list)):
            df_list[i][col] = df_list[i][col].apply(lambda x: x if x in common else -20)
    # 标称型缺失值统一为-30
    for df in df_list:
        df[nominal_type] = df[nominal_type].fillna(-30)
    # 数值型特征缺失值补充
    df_list = medium_fill(df_list, numerical_type)

    # 特征选择
    df_list = var_delete(df_list, nominal_type, numerical_type)  # 方差选择法
    df_list = select_best_chi2(df_list)  # 卡方验证选择法
    # # 标称型独热编码
    # columns = df_list[2].columns.values.tolist()  # 测试集不包含label
    # new_nominal_type = list(set(columns) & set(nominal_type))  # 新的标称型列
    # new_numerical_type = list(set(columns) & set(numerical_type))  # 新的数值型列
    # df_list, new_nominal_type = one_hot_encode(df_list, new_nominal_type)  # 独热编码
    # for df in df_list:
    #     print(df.head(5))
    # print(new_numerical_type)
    # print(new_nominal_type)
    # new_feature = new_nominal_type + new_numerical_type  # 新的特征列
    #
    # # PCA降维
    # a = df_list[0][new_feature]
    # n_components = 30  # pca的维度
    # new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
    # pca_train_set = pd.DataFrame(pca_decompose(df_list[0][new_feature].as_matrix(), n_components), columns=new_pridictors)  # DF转矩阵
    # pca_valid_set = pd.DataFrame(pca_decompose(df_list[1][new_feature].as_matrix(), n_components), columns=new_pridictors)
    # pca_test_set = pd.DataFrame(pca_decompose(df_list[2][new_feature].as_matrix(), n_components), columns=new_pridictors)  # DF转矩阵
    # # 将相应的id补上
    # pca_train_set['label'] = df_list[0]['label']
    # pca_train_set['pid'] = train_set['pid']
    # pca_train_set['fid'] = train_set['fid']
    #
    # pca_valid_set['label'] = df_list[1]['label']
    # pca_valid_set['pid'] = valid_set['pid']
    # pca_valid_set['fid'] = valid_set['fid']
    #
    # pca_test_set['pid'] = test_set['pid']
    # pca_test_set['fid'] = test_set['fid']

    # 建模
    columns = df_list[2].columns.values.tolist()  # 测试集不包含label
    new_train_set = df_list[0]
    new_train_set['pid'] = train_set['pid']
    new_train_set['fid'] = train_set['fid']
    new_valid_set = df_list[1]
    new_valid_set['pid'] = valid_set['pid']
    new_valid_set['fid'] = valid_set['fid']
    new_test_set = df_list[2]
    new_test_set['pid'] = test_set['pid']
    new_test_set['fid'] = test_set['fid']

    # 训练
    x = new_train_set[columns]
    y = new_train_set['label']
    model = check_model(x, y)

    # 验证
    x_valid = valid_set[columns]
    y_valid = valid_set['label']
    y_pred = model.predict(x_valid)
    y_pred_prob = model.predict_proba(x_valid)[:, 1]
    print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
    print('AUC Score (Train): %.4g' % roc_auc_score(y_valid, y_pred_prob))

    # 测试
    test_x = test_set[columns]
    test_y_pred_prob = model.predict_proba(test_x)[:, 1]
    test_y_label = model.predict(test_x)
    print(test_y_label)
    test_set['pred'] = test_y_pred_prob
    test_set['label'] = test_y_label
    summit_file = test_set[['pid', 'fid', 'pred', 'label']].copy()
    summit_file.columns = ['pid', 'fid', 'pred', 'label']
    summit_file.to_csv('summit.csv', columns=['pid', 'fid', 'pred', 'label'], index=False)  # 不要索引, header=False 不要列头

    # 保存模型
    save_model(model)


def get_fina_result():
    """
    生成最终流失的1500个fid
    :return:
    """
    get_model()
    result = pd.read_csv('summit.csv')
    result = result[['fid', 'pred']].copy()
    result['count'] = 1
    temp = result.groupby(['fid'], as_index=False).count()  # 统计每个fid出现的次数
    res = (result.groupby(['fid'], as_index=False)['pred'].sum()).copy()  # 基于fid求和流失概率
    res['fid_count'] = temp['count']
    res['final_probability'] = res.apply(lambda x: x['pred'] / x['fid_count'], axis=1)
    res = (res.sort_values('final_probability', ascending=False)).copy()  # 降序排序
    res.to_csv('final_disappear_fid.csv', columns=['fid', 'final_probability'], index=False)


if __name__ == '__main__':
    # process_data()
    get_fina_result()

    # x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 1]])
    # x1 = StandardScaler().fit_transform(x)
    # print(x1)
    # print(len(numerical_type) + len(nominal_type))

    # columns = train_set.columns.values.tolist()
    # for i in range(len(columns) - 1):
    #     col = columns[i]
    #     print(col, train_set[col].unique())
    #     print(col, valid_set[col].unique())
    #     print(col, test_set[col].unique())
    #     print('--------------')

    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # print(iris.data)
    # model1 = SelectKBest(chi2, k=2)  # 选择k个最佳特征
    # model1.fit_transform(iris.data, iris.target)
    # # print(columns)
    # # print(x)
    # columns = ['a', 'b', 'c', 'd']
    # index = model1.get_support(indices=True)
    # print(index)


