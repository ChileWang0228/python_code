#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-3-12 21:19
@Author:ChileWang
@algorithm：
暂时用不到的函数
"""
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# def select_best_k_for_kmeans(more_class_set):
#     """
#     找到最佳的k均值分类
#     :param more_class_set:多数类样本
#     :return:
#     """
#     print(more_class_set[numerical_type].head(5))
#     k_list = range(1, 10)
#     mean_num_list = []  # 装载每个数值型特征原本的平均数
#     std_num_list = []  # 装载每个数值型特征原本的标准差
#     for num_t in numerical_type:
#         mean_num = more_class_set[num_t].mean()  # 该列平均值
#         mean_num_list.append(mean_num)
#         std_num = more_class_set[num_t].std()  # 该列标准差
#         std_num_list.append(std_num)
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: 1.0 * (x-mean_num)/std_num)  # 数值型每一列进行数据标准化
#
#     mean_distortions = []  # 计算所有点与对应聚类中心的距离的平方和的均值
#     for k in k_list:
#         kmeans = KMeans(n_clusters=k, n_jobs=-1)
#         kmeans.fit(more_class_set)
#         # 解释此处代码
#         # cdist:Computes distance between each pair of the two collections of inputs.
#         # 理解为计算某个与其所属类聚中心的欧式距离
#         # 最终是计算所有点与对应中心的距离的平方和的均值
#         mean_distortions.append(sum(np.min(cdist(more_class_set, kmeans.cluster_centers_, 'euclidean'),
#                                            axis=1))/more_class_set.shape[0])
#     draw_polygonal_line(k_list, mean_distortions, title='Selecting k with the Elbow Method',
#                         color='bx-', x_label='K', y_label='Average Dispersion')
#
#     for i in range(len(numerical_type)):
#         num_t = numerical_type[i]
#         mean_num = mean_num_list[i]  # 该列原本平均值
#         std_num = std_num_list[i]  # 该列原本标准差
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: round(x * std_num + mean_num, 1))  # 复原数值型每一行数据
#     more_class_set[numerical_type] = more_class_set[numerical_type].replace(-0.0, 0.0)  # replace函数是右边替换左边
#     print(more_class_set[numerical_type].head(5))

# def sample_equilibrium(k, train_set, feature):
#     """
#     :param k: 最合适的累簇数量
#     :param train_set: 训练集
#     :param feature: 用来聚类的特征
#     :return: 均衡的样本
#     """
#     mean_num_list = []  # 装载每个数值型特征原本的平均数
#     std_num_list = []  # 装载每个数值型特征原本的标准差
#     more_class_set = train_set[train_set['label'] == 0].copy()  # 多数类
#     more_class_set.reset_index(inplace=True)  # 重新设置索引
#     less_class_set = train_set[train_set['label'] == 1].copy()  # 少数类
#     less_class_set.reset_index(inplace=True)
#
#     # 对多数类数值型每一列进行数据标准化
#     for num_t in numerical_type:
#         mean_num = more_class_set[num_t].mean()  # 该列平均值
#         mean_num_list.append(mean_num)
#         std_num = more_class_set[num_t].std()  # 该列标准差
#         std_num_list.append(std_num)
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: 1.0 * (x - mean_num) / std_num)
#
#     # k-means聚类
#     print('---------K-means-------------')
#     model = KMeans(n_clusters=k, n_jobs=-1, random_state=9)
#     model.fit(more_class_set[feature])
#     r1 = pd.Series(model.labels_).value_counts()  # 统计每个类别的数目
#     r2 = pd.DataFrame(model.cluster_centers_)  # 找出每个类别的聚类中心
#     r = pd.concat([r2, r1], axis=1)  # 横向链接(0是纵向),得到聚类中心对应的类别下的数目
#     # print(r)
#
#     kmeans_df = []  # 多数类聚类后的数据集列表
#     for i in range(k):
#         index = np.where(model.labels_ == i)  # 取出属于该簇类的所有索引
#
#         kmeans_df.append(more_class_set.loc[index].copy())
#
#     # 对多数类数值型进行复原
#     for c_df in kmeans_df:
#         for i in range(len(mean_num_list)):
#             mean_num = mean_num_list[i]
#             std_num = std_num_list[i]
#             num_t = numerical_type[i]
#             c_df[num_t] = c_df[num_t].apply(lambda x: round(x * std_num + mean_num, 1))  # 复原数值型每一行数据
#         c_df[numerical_type] = c_df[numerical_type].replace(-0.0, 0.0)  # replace函数是右边替换左边
#
#     train_set_list = []  # 将聚类后的多数类集合分别和少数类合成一个新训练集，再将其装载在该列表
#     for i in range(k):
#         print(kmeans_df[i].shape[0] / less_class_set.shape[0])
#         print('---------------')
#         train_set = pd.concat([kmeans_df[i], less_class_set], axis=0)  # 纵向链接(１是横向)
#         train_set_list.append(train_set)
#
#     return train_set_list
