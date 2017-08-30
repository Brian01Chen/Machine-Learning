import csv
import os,copy
import random
import math

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


    
#分层抽样法
#对不同的类别的数据都进行随机抽样
def group_splitDataset(dataset,splitRatio):
    bucket1 = [ data for data in dataset if data[-1] == 0]
    bucket2 = [ data for data in dataset if data[-1] == 1]
    
    trainSize1 = int(len(bucket1) * splitRatio)
    trainSize2 = int(len(bucket2) * splitRatio)
    trainSet1,testSet1,trainSet2,testSet2,testSet = [],[],[],[],[]
   
    copy1 = list(bucket1)
    copy2 = list(bucket2)
    while len(trainSet1) < trainSize1:
        index = random.randrange(len(copy1))
        trainSet1.append(copy1.pop(index))
    while len(trainSet2) < trainSize2:
        index = random.randrange(len(copy2))
        trainSet2.append(copy2.pop(index))
    testSet = copy1 + copy2
    return [['Mutiple-Layer',trainSet1,trainSet2],testSet]

# 求数学期望
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# 求样本方差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#中位数
def median(numbers):
    sortnumbers = sorted(numbers)
    length = len(sortnumbers)
    #当列表有奇数个元素，返回中间的元素
    if length % 2 == 1:
        return sortnumbers[int (((length + 1)/2)-1)]
    else:
    #当列表有偶数个元素，返回中间2个元素的均值
        v1 = sortnumbers[int(length/2)]
        v2 = sortnumbers[int(length/2)-1]
        return (v1+v2)/2.00

def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    #summaries = [(median(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# 对数据进行标准化，这样不同的attribute的数据都将按同一标准计算。
# 标准化的方式： ele' = (ele - mean)/stdev     ele' 的期望和方差 分别为0,1  
def standard_dateset(dataset):
    
    std_dataset = copy.deepcopy(dataset)
    def f_std_data(vector):
        summaries = summarize(dataset)
        try:
            vector = [(vec-summaries[num][0])/summaries[num][1] if num != len(vector)-1 else vec for num,vec in enumerate(vector)]
        except IndexError:
            print ( 'vector attribute too long,out of range' )
        return vector
    return list(map(f_std_data,std_dataset))

'''
def standard_dateset_old(dataset):
    summaries = summarize(dataset)
    std_dataset = copy.deepcopy(dataset)
    for i in std_dataset:
        for j in range(0,len(i)-1):
            i[j] = (i[j]-summaries[j][0])/summaries[j][1]
    return std_dataset
'''


# KNN K临近算法
def calc_k_distance(trainingSet,testSet,k):
    # 数据集标准化
    if trainingSet[0] == 'Mutiple-Layer':
       trainingSet1 = standard_dateset(trainingSet[1])
       trainingSet2 = standard_dateset(trainingSet[2])
       trainingSet = trainingSet1 + trainingSet2
    else:
       trainingSet = standard_dateset(trainingSet) 
    vector = standard_dateset(testSet)
    
    predictions = []
    for v in vector:
        distance_compare = []
        for i in trainingSet:
            couple = [ j for j in zip(i,v)]
            del couple[-1]
            #计算曼哈顿距离
            distance = sum(abs(j[0]-j[1]) for j in couple)
            distance_compare.append((distance,i[-1]))
        #按距离排序,按距离默认升序排列，取 前K个与 目标向量Vector相近的数据
        distance_compare = sorted(distance_compare,key = lambda distance_compare:distance_compare[0])
        #算权重，距离越小，权重越高。 权重计算方式按距离的倒数算。
        #总group 距离 sum(1/d(group))/sum(1/d) 
        weight_distance = sum(1/d[0] for d in distance_compare[0:k])
        v1,v2 = 0 , 0
        for d in distance_compare[0:k]:
            if d[-1] == 0:
                v1 += 1/d[0]/weight_distance
            else:
                v2 += 1/d[0]/weight_distance
        
        if v1> v2:
            predict_value = 0
        else:
            predict_value = 1
        predictions.append(predict_value)
                    
    return predictions
        
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct /float(len(testSet))) * 100.0 
    
if __name__ == '__main__':
    path = r'C:\Users\IBM_ADMIN\AppData\Local\Programs\Python\Python35\mydata\pima-indians-diabetes'
    # data get from https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
    filename = r'data.csv'
    fullname = os.path.join(path,filename)
    dataset = loadCsv(fullname)
    splitRatio = 0.8

    #简单随机采样方式准确率
    trainingSet , testSet = splitDataset(dataset,splitRatio)
    predictions = calc_k_distance(trainingSet,testSet,4)
    accuracy = getAccuracy(testSet, predictions)
    print (('Accuracy: {0}%').format(accuracy))
    
    #分层随机采样方式准确率
    trainingSet , testSet = group_splitDataset(dataset,splitRatio)
    predictions = calc_k_distance(trainingSet,testSet,4)
    accuracy = getAccuracy(testSet, predictions)
    print (('Accuracy: {0}%').format(accuracy))
    # 10次结果，当前数据集 简单随机采样准确率>分层随机采样准确率

    
