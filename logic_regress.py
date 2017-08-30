import csv
import os,copy
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
    lines = csv.reader(open(filename,'r'))
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

#随机抽样
def splitDataset(dataset,splitRatio):

    trainSize = int(len(dataset) * splitRatio)
    trainSet,testSet = [],[]
    traingroup = []
    copy = list(dataset)
    
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    testSet = copy
    
    return [trainSet,testSet]

def sigmoid(num):
    return 1.0 / (1+ math.exp(-num))

def Logical_Cost_Function(train_row,weight):
    #假设回归方程为多元一次线性回归, weight 可以简单的看做方程的参数，比如 [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    
    #x = [1,train_row[0],train_row[1],train_row[2],train_row[3],train_row[4],train_row[5],train_row[6],train_row[7]]
    #train_row[8] 为分类值 不计入 x
    x = [1] + train_row[:-1]

    #y = weight[0] + weight[1]*train_row[0] + weight[2]*train_row[1] + weight[3]*train_row[2] + weight[4]*train_row[3] \
    #     + weight[5]*train_row[4] + weight[6]*train_row[5] + weight[7]*train_row[6] + weight[8]*train_row[7]
    y = sum(list(map(lambda a,b : a*b , x, weight)))

    #x:train_row[i](i=0,1...7)代表数据的分类值的观测值，train_row[8]表示数据的分类值,1 代表患有糖尿病
    #b表示方程的参数向量
    #y表示方程的y值，意义为通过线性方程拟合分类值的计算值。

    #极大似然函数
    cost = float(math.pow(sigmoid(y),train_row[8]))*float(math.pow((1-sigmoid(y)),(1-train_row[8])))
    #print ('%s : %d' % ("似然函数值：" ,cost))
    #对数极大似然函数,针对参数b0,b1...b8
    try:
        log_cost = float(train_row[8]*y-math.log(1+math.exp(y)))
    except Exception as e:
        print (e)
        print (train_row,y)
    #print ('%s : %d' % ("对数极大似然函数：" , log_cost))
    #分别对 上面对数极大似然函数 自变量(即x[j])  求导
    log_cost_delta = []
    for j in range(len(weight)):

        log_cost_delta_j = float(train_row[8] - sigmoid(y))* x[j]
        log_cost_delta.append(log_cost_delta_j)

    #print ('%s : %d' % ("对数极大似然函数对b求导：" , log_cost_delta))
    return log_cost,log_cost_delta
    
def trainLogical(trainSet,opt):
    weight,alpha,maxstep = opt[0],opt[1],opt[2]

    trainSet = standard_dateset(trainSet)

    min_cost = None
    for j in range(maxstep):

        current_cost,current_row_cost = 0,0
        cost_delta, cost_row_delta = [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for train_data in trainSet:
            current_row_cost, cost_row_delta = Logical_Cost_Function(train_data, weight)
            #
            cost_delta = [sum(x) for x in zip(*[cost_delta, cost_row_delta])]
            #当前的cost
            current_cost += current_row_cost
        #方程参数weight计算
        weight = list(np.array(weight) + alpha * np.array(cost_delta))
        #将第一次计算所得cost赋予最小cost,否则，只有当当前的cost小于最小cost 才会继续迭代。
        print (current_cost)
        if min_cost is None or abs(current_cost) <= abs(min_cost):
            min_cost = current_cost
        else:
            print (j)
            break
    return weight

def testLogical(testSet,weight):

    total_num,right_num = 0,0
    testSet = standard_dateset(testSet)
    print (weight)
    for test_row in testSet:
        total_num += 1
        x = [1] + test_row[:-1]
        y = sum(list(map(lambda a, b: a * b, x, weight)))
        predict = sigmoid(y) > 0.5
        if predict == bool(test_row[-1]) :
            right_num += 1

    accuracy = float(right_num) / total_num

    return accuracy

if __name__ == '__main__':
    path = r'C:\Users\IBM_ADMIN\AppData\Local\Programs\Python\Python35\mydata\pima-indians-diabetes'
    # data get from https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
    filename = r'data.csv'
    fullname = os.path.join(path,filename)
    dataset = loadCsv(fullname)
    splitRatio = 0.8

    #简单随机采样
    trainingSet , testSet = splitDataset(dataset,splitRatio)
    b = [ 1 for i in range(9)]
    alpha, maxstep = 0.005,200
    opt = [b,alpha,maxstep]
    weight = trainLogical(trainingSet,opt)
    accuracy = testLogical(testSet,weight)
    print (accuracy)
        
    




    
    
