import xlrd

import numpy as np
import random

from sklearn.cross_validation import train_test_split


def loadData():
    excelRead = xlrd.open_workbook('2014 and 2015 CSM dataset.xlsx')
    sheet = excelRead.sheet_by_name('Sheet1')

    Ratings = sheet.col_values(2)
    Ratings = Ratings[1:]  # 只取数据

    dataSet = []
    for i in range(sheet.nrows):
        if i == 0 : continue
        else:
            dataSet.append(sheet.row_values(i))
    return dataSet,Ratings

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


if __name__ == '__main__':
    dataSet, Ratings = loadData()
    trainSet, testSet = train_test_split(dataSet, test_size=0.2)  # 划分训练集和测试集
    # 5-13 列 有效数据
