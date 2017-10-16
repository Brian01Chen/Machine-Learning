import csv
import os
import random
import math

# load 数据集
def loadCsv(filename):
    lines = csv.reader(open(filename,'r'))
    dataset = list(lines)

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# 使用random模块随机从Dataset中取数据.随机的策略: 生成 [0,len(copy(dataset))] 的均匀分布
# 并按SplitRatio划分数据集成训练集和测试集。
def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet,testSet = [],[]
    traingroup = []
    copy = list(dataset)
    
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        #非均匀分布，抽样方式为无重复的简单随机抽样！
        trainSet.append(copy.pop(index))
        
        #有放回/有重复的简单随机抽样！
        #trainSet.append(copy[index])
        #traingroup.append(index)  
    testSet = copy
    #testSet  = [ dataset[record] for record  in range(0,len(dataset)) if record not in traingroup]
    print (len(testSet))
    return [trainSet,testSet]


#按最后预测数据的结果分类，本次数据预测是否糖尿病，只有2类。
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

#重复抽样的标准差/不重复抽样的标准差。后者需要乘 sqr(N-n/N-1) 的系数，更小。
#当前采用的重复抽样的标准差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#获取样本的特征值
def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]#zip过的列表对象解压
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


#  强假设，假设样本的attribute值 数据是按正态分布呈现的。
#  比如第8列 age, 每个样本的数据是按正态分布的
def calculateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent #正态分布的密度函数


#通过计算不同类别的 样本的均值和样本的方差，可以算出
#不同类别的似然概率P(x(观测值,即测试集的当前数据的年龄)|类别的样本均值,类别的样本方差)
def calculateClassProbabilities(summaries,inputVector):
    probabilities = {}

    for classValue, classSummaries in summaries.items():
        
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,stdev)
    return probabilities


#按最大概率判断当年的数据是属于哪个分类
def predict(summaries,inputVector):
    probabilities = calculateClassProbabilities(summaries,inputVector)
    bestLabel, bestProb = None , -1
    #print (probabilities)
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
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
    splitRatio = 0.67
    trainingSet , testSet = splitDataset(dataset,splitRatio)
    gini = calcGini(trainingSet)
    entropy = calcEnt(trainingSet)
    print (gini)
    print (entropy)

    #summaries = summarizeByClass(trainingSet)
    #predictions = getPredictions (summaries,testSet)
    #accuracy = getAccuracy(testSet, predictions)
    #print (('Accuracy: {0}%').format(accuracy))
