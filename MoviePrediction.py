
import pandas as pd
import numpy as np
import random

from sklearn.cross_validation import train_test_split

def loadData():
    df = pd.read_excel('2014 and 2015 CSM dataset.xlsx')
    dataSetDF = df[
        ['Ratings', 'Budget', 'Screens', 'Sequel']
    ]
    # 'Aggregate Followers''Genre',  ,'Sentiment' 'Views', 'Likes', 'Dislikes', 'Comments'
    dataSetDF = dataSetDF.fillna(dataSetDF.mean())
    dataSetArr = np.array(dataSetDF)  # np.ndarray()
    dataSet = dataSetArr.tolist()  # list
    return dataSet

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化										#存储每次更新的回归系数
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights


def classifyVector(inX, weights):
    prob = 10* sigmoid(sum(inX*weights))
    return prob

if __name__ == '__main__':
	dataSet = loadData() # 5-13 列 有效数据
	trainSet, testSet = train_test_split(dataSet, test_size=0.2)
	trainRatings = [];
	testRatings = []
	for data in trainSet:
		trainRatings.append(data[0])
		del (data[0])
	for data in testSet:
		testRatings.append(data[0])
		del (data[0])
	trainArr = np.array(trainSet)
	trainRarr = np.array(trainRatings)
	trainWeights = stocGradAscent1(trainArr, trainRarr, 100)
	print(trainWeights)
	print(classifyVector(np.array(testSet[0]), trainWeights))
	print(testRatings[0])
