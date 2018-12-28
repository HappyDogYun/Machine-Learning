#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle as pk
import numpy as np

Absolute_path = os.path.realpath(__file__)              # 获取当前.py文件的绝对路径
file_path = os.path.split(Absolute_path)[0]             # 将路径去除文件名称，只保留地址
train_path = os.path.join(file_path, "train.data")      # 生成训练集文件路径
test_path = os.path.join(file_path, "test.data")        # 生成测试集文件路径
dict_path = os.path.join(file_path, "dictionary.data")  # 生成字典文件路径

# 全局变量定义，为了最大限度的节省计算资源
global prob_matrix      # 概率矩阵，记录了所有分类中出现的单词频率，即条件概率
global prob_default
global priori_pr        # 计算得出的先验概率，用于最后的计算

prob_matrix = {}
prob_default = {}
priori_pr = [0.0, 0.0, 0.0, 0.0]


def generate_word_dict():
    my_dict = {}                        # 自定义的字典结构，key是分类值，value是一个单词词频字典
    count = [0, 0, 0, 0]                # 用来记录每一类新闻的总数
    with open(train_path, "rb") as f:
        train_data = pk.load(f)
        f.close()                       # 关闭文件
    for article in train_data:
        content = article[0]            # 抽取新闻内容
        sort = article[1]               # 抽取新闻分类

        if sort not in my_dict.keys():
            my_dict[sort] = {}          # 构造字典的结构

        if sort == 0:                   # 计算每一类新闻的总数
            count[0] = count[0] + 1
        if sort == 1:
            count[1] = count[1] + 1
        if sort == 2:
            count[2] = count[2] + 1
        if sort == 3:
            count[3] = count[3] + 1

        for word in content:            # 构造字典的value结构
            if word not in my_dict[sort].keys():
                my_dict[sort][word] = 1
            else:
                my_dict[sort][word] = my_dict[sort][word] + 1

    with open(dict_path, "wb") as f:    # 将字典以wb的方式保存
        pk.dump(my_dict, f)
        f.close()                       # 关闭文件流

    total = 0                           # 所有的新闻文本数量，用于计算先验概率
    for i in range(0, 4):
        total = total + count[i]

    global priori_pr                    # 计算先验概率
    for i in range(0, 4):
        priori_pr[i] = float(count[i]) / float(total)

    global prob_matrix
    global prob_default
    word_num = [0, 0, 0, 0]             # 用来存储每一类新闻中的单词总数
    for sort, words_list in my_dict.items():
        for count in words_list.values():
            word_num[sort] = word_num[sort] + count
        prob_matrix[sort] = {}
        for word in words_list.keys():  # 生成概率矩阵
            prob_matrix[sort][word] = float(my_dict[sort][word] / word_num[sort]) * total
        prob_default[sort] = float(1 * total / word_num[sort])  # 避免出现概率为0的情况，其实并不是正确的平滑处理


def naive_bayes_classifier(news, my_dict):
    global prob_matrix
    global prob_default
    global priori_pr

    pre_sort = {}                                   # 计算得出属于各类的概率
    for sort in prob_matrix.keys():
        pre_sort[sort] = priori_pr[sort]            # 先乘以先验概率
        for word in news:
            if word in prob_matrix[sort].keys():    # 类乘计算
                pre_sort[sort] = float(pre_sort[sort] * prob_matrix[sort][word])
            else:
                pre_sort[sort] = float(pre_sort[sort] * prob_default[sort])
    return np.argmax(list(pre_sort.values()))       # 返回概率最大值所属于的分类数


def test():
    generate_word_dict()                            # 为了最大限度的保持test函数不改动，只能这样做了
    with open(test_path, "rb") as f:
        test_data = pk.load(f)
        f.close()
    try:
        with open(dict_path, "rb") as f:
            word_dict = pk.load(f)
            f.close()
    except:
        print("你的字典没有完成或者出了一些问题，请修改！")
        return
    tp = np.zeros(4)
    fp = np.zeros(4)
    try:
        for data in test_data:
            your_ans = naive_bayes_classifier(data[0], word_dict)
            if your_ans == data[1]:
                tp[data[1]] += 1
            else:
                fp[your_ans] += 1
    except :
        print("你的分类器没有完成或者出了一些问题，请修改！")
        return
    p = tp / (tp + fp)
    r = tp / 200
    f = 2 * p * r / (p + r)
    print("__" * 30)
    for i, j in enumerate(['energy', 'estate', 'sports', 'entertainment']):
        print("对于", j, "的分类情况为:")
        print("准确度: ", p[i], "  召回率：", r[i], "  F值：", f[i])
    print("__"*30)
    return


if __name__ == "__main__":
    if len(sys.argv) != 1:
        if sys.argv[1] == 'train':
            print('正在生成字典（训练）...')
            try:
                generate_word_dict()
            except:
                print('generate_word_dict函数出了一些bug，请修改！')
            else:
                print('训练完成！')
    else:
        test()
