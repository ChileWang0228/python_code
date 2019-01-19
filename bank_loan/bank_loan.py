#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编译环境：Anaconda python 3.7.2
@Created on 2019-1-16 21:52
@Author:ChileWang
@algorithm：
本练习赛的数据，选自UCI机器学习库中的「银行营销数据集(Bank Marketing Data Set)
这些数据与葡萄牙银行机构的营销活动相关。这些营销活动以电话为基础，一般，
银行的客服人员需要联系客户至少一次，以此确认客户是否将认购该银行的产品（定期存款）。
因此，与该数据集对应的任务是「分类任务」，「分类目标」是预测客户是(' 1 ')或者否(' 0 ')购买该银行的产品。
"""
import os
import pickle
import pandas as pd
import numpy as np
from numpy import linalg as la
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import decomposition
import lightgbm as lgb

pd.set_option('display.max_columns', None)  # 显示所有列


def one_hot_encode(encode_df, encode_list):
    """
    对属性进行独热编码
    :param encode_df: 数据集
    :param encode_list: 需要独热编码的列
    :return:
    """
    feature = []
    for col in encode_list:
        one_hot_columns = encode_df[col].unique()  # 独热编码后的列
        temp_df = pd.get_dummies(encode_df[col].replace('null', np.nan))  # 独热编码
        temp_df.columns = one_hot_columns
        encode_df[one_hot_columns] = temp_df  # 合并到原来的数据集
        feature.extend(one_hot_columns)  # 新生成的特征
    return encode_df, feature


def process_data(df):
    """
    将数据进行数值化处理
    :return:
    """

    # test_list = df_train['marital'].replace('null', np.nan)
    # 需要进行独热编码的列
    one_hot_encode_columns = ['job', 'marital', 'education', 'default', 'housing',
                              'loan', 'contact', 'month', 'poutcome']  # 类别特征
    predictors = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']  # 数值特征
    for fea in one_hot_encode_columns:
        df[fea] = df[fea].astype('category')
        print(df[fea].head())
    for fea in predictors:
        df[fea] = df[fea].astype('float')
    # encoded_df, feature = one_hot_encode(df, one_hot_encode_columns)  # 独热编码
    # predictors += feature
    # return encoded_df, predictors
    predictors += one_hot_encode_columns  # 总特征
    return df, predictors


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


def svd_decompose(data_mat):
    """
    SVD分解降维
    :param data_mat:
    :return:
    """
    u, s, vt = la.svd(data_mat)
    return u, s, vt


def cal_percentage(sigma, percentage=0.9):
    """
    设定前ｎ个奇异值能量的占比，求出需要分解的Ｋ
    :return:
    """
    print(sigma)
    sum_energy = sum(sigma**2)  # 总能量
    squre = sigma**2  # 奇异值求平方
    for i in range(1, len(sigma) + 1):
        des_energy = sum(squre[:i])  # 前ｉ个奇异值的能量
        if des_energy / sum_energy >= percentage:
            k = i + 1
            return k
    return np.shape(sigma)[0]


def build_new_mat(data_mat):
    """
    构造新的svd分解矩阵
    :param data_mat: 原始矩阵
    :return:
    """
    u, s, vt = svd_decompose(data_mat)
    k = cal_percentage(s)
    new_sigma = np.mat(np.zeros((k, k)))  # 生成全０矩阵
    print(new_sigma)
    for i in range(k):
        new_sigma[i][i] = s[i]

    new_mat = u[:, :k] * new_sigma * vt[:k, :]
    print(new_mat)
    return u[:, :k], new_sigma, vt[:k, :], new_mat


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
                            cv=10,  # 五折交叉验证
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
    with open('bank_loan_modle', 'rb') as f:
        model = pickle.load(f)
    return model


def pre_process_data():
    """
    数据预处理及相关信息可视化
    :return:
    """
    df_train = pd.read_csv('train_set.csv', keep_default_na=False)  # 训练集
    df_test = pd.read_csv('test_set.csv', keep_default_na=False)  # 测试集合
    print('负样本:', df_train[df_train['y'] == 0].shape[0])
    print('正样本:', df_train[df_train['y'] == 1].shape[0])
    print('总样本:', df_train['y'].shape[0])
    train_set, predictors = process_data(df_train)  # 经过数值化处理的训练数据集
    test_set = process_data(df_test)[0]  # 经过数值化处理的测试数据集
    # print(train_set[predictors].as_matrix())
    # n_components = 15  # pca的维度
    # new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
    # pca_train_set = pd.DataFrame(pca_decompose(train_set[predictors].as_matrix(), n_components), columns=new_pridictors)  # DF转矩阵
    # pca_test_set = pd.DataFrame(pca_decompose(test_set[predictors].as_matrix(), n_components), columns=new_pridictors)  # DF转矩阵
    # pca_train_set['y'] = train_set['y']
    # pca_train_set['ID'] = train_set['ID']
    # pca_test_set['ID'] = test_set['ID']
    train_set, valid_set = train_test_split(train_set, test_size=0.2, stratify=train_set['y'], random_state=100)
    # train_set, valid_set = train_test_split(pca_train_set, test_size=0.2, stratify=pca_train_set['y'], random_state=100)
    return train_set, valid_set, test_set, predictors

    # return train_set, valid_set, pca_test_set, new_pridictors


def main():
    """
    模型训练及预测结果
    :return:
    """
    # train_set, valid_set, test_set, predictors = pre_process_data()
    train_set, valid_set, test_set, predictors = pre_process_data()
    # 训练集
    x = train_set[predictors]
    y = train_set['y']
    # gbm0 = GradientBoostingClassifier(random_state=10)
    # 随机数种子没有设置 random.seed()，每次取得的结果就不一样，它的随机数种子与当前系统时间有关。
    model = check_model(x, y)

    # 验证集
    x_valid = valid_set[predictors]
    y_valid = valid_set['y']
    y_pred = model.predict(x_valid)
    y_pred_prob = model.predict_proba(x_valid)[:, 1]
    print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
    print('AUC Score (Train): %.4g' % roc_auc_score(y_valid, y_pred_prob))

    # 测试集
    test_x = test_set[predictors]
    test_y_pred_prob = model.predict_proba(test_x)[:, 1]
    test_set['pred'] = test_y_pred_prob
    summit_file = test_set[['ID', 'pred']].copy()
    summit_file.columns = ['ID', 'pred']
    summit_file.to_csv('summit.csv', columns=['ID', 'pred'], index=False)  # 不要索引, header=False 不要列头

    # 保存模型
    save_model(model)


if __name__ == '__main__':
    main()
    # pre_process_data()