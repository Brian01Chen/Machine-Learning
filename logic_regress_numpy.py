import csv
import os, copy
import random
import math
import numpy as np

'''
DataSet

#1. 怀孕次数；
#2. 口服葡萄糖耐量实验两小时后的血浆葡萄糖浓度；
#3. 舒张压（mm Hg）；
#4. 三头肌皮褶厚度（mm）；
#5. 血清胰岛素（mu U/ml）；
#6. 身体质量指数（BMI）：体重（公斤）除以身高（米）的平方；
#7. 糖尿病家谱；
#8. 年龄（岁）。

6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
'''


##
# load 数据集
def loadCsv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# 求数学期望
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# 求样本方差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# 中位数
def median(numbers):
    sortnumbers = sorted(numbers)
    length = len(sortnumbers)
    # 当列表有奇数个元素，返回中间的元素
    if length % 2 == 1:
        return sortnumbers[int(((length + 1) / 2) - 1)]
    else:
        # 当列表有偶数个元素，返回中间2个元素的均值
        v1 = sortnumbers[int(length / 2)]
        v2 = sortnumbers[int(length / 2) - 1]
        return (v1 + v2) / 2.00


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    # summaries = [(median(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# 对数据进行标准化，这样不同的attribute的数据都将按同一标准计算。
# 标准化的方式： ele' = (ele - mean)/stdev     ele' 的期望和方差 分别为0,1
def standard_dateset(dataset):
    std_dataset = copy.deepcopy(dataset)

    def f_std_data(vector):
        summaries = summarize(dataset)
        try:
            vector = [(vec - summaries[num][0]) / summaries[num][1] if num != len(vector) - 1 else vec
                      for num, vec in enumerate(vector)]
        except IndexError:
            print('vector attribute too long,out of range')
        return vector

    return list(map(f_std_data, std_dataset))


# 随机抽样
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet, testSet = [], []
    traingroup = []
    copy = list(dataset)

    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    testSet = copy

    return [trainSet, testSet]

def sigmoid(y):
    return 1.0/(1+ np.exp(-y))

def trainLogic(dataset,ops):
    train_x = []
    train_y = []
    alpha = ops[0]
    weights = ops[1]
    step = ops[2]
    for data_row in dataset:
        row = [1.0]
        row += data_row[:-1]
        train_x.append(row)
        train_y.append(data_row[-1])
    
    X = np.mat(train_x)
    Y = np.mat(train_y).transpose()
    for k in range(step):
        err = Y - sigmoid(X*weights)
        weights = weights + alpha * X.transpose() * err
    print (weights)

if __name__ == '__main__':
    path = r'C:\Users\IBM_ADMIN\AppData\Local\Programs\Python\Python35\mydata\pima-indians-diabetes'
    # data get from https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
    filename = r'data.csv'
    fullname = os.path.join(path, filename)
    dataset = loadCsv(fullname)
    splitRatio = 0.8

    # 简单随机采样
    trainSet, testSet = splitDataset(dataset, splitRatio)
    ops = []
    alpha = 0.01
    ops.append(alpha)
    print (np.shape(np.ones(np.shape(trainSet)[1])))
    print (np.shape(np.ones((np.shape(trainSet)[1],1))))
    ops.append(np.ones((np.shape(trainSet)[1],1)))
    ops.append(100)
    print (ops)
    trainLogic(standard_dateset(dataset),ops)


