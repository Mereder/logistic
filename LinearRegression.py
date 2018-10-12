import xlrd
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd


def loadData():
    df = pd.read_excel('2014 and 2015 CSM dataset.xlsx')
    dataSetDF = df[
        ['Ratings', 'Genre', 'Budget', 'Screens', 'Sequel', 'Sentiment', 'Views', 'Likes', 'Dislikes', 'Comments',
         'Aggregate Followers']]
    dataSetDF = dataSetDF.fillna(dataSetDF.mean())
    dataSetArr = np.array(dataSetDF)  # np.ndarray()
    dataSet = dataSetArr.tolist()  # list
    return dataSet

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    ws = xTx.I * (xMat.T * yMat)  # 求 w=(x.T*x).I*x.T*y
    return ws

def batchGradientDescent(maxiter,x,y,theta,m):
    alpha = 0.01
    xTrains = x.transpose()
    for i in range(0, maxiter):
        hypothesis = np.dot(x, theta)
        loss = (hypothesis - y)
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
        cost = 1.0 / 2 * m * np.sum(np.square(np.dot(x, np.transpose(theta)) - y))
        print("cost: %f" % cost)
    return theta

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


if __name__ == '__main__':
    # 去掉了前两列以及 Gross的数据
    dataSet = loadData()
    # 来处理训练集和测试集
    trainSet, testSet = train_test_split(dataSet, test_size=0.2)
    trainRatings = [];testRatings = []
    for data in trainSet:
        trainRatings.append(data[0])
        del (data[0])
    for data in testSet:
        testRatings.append(data[0])
        del (data[0])
    trainArr = np.array(trainSet)
    trainRarr = np.array(trainRatings)

    m,n = np.shape(trainArr)
    theta = np.ones(n)

    # result = batchGradientDescent(10000, trainArr, trainRarr, theta, 0.01)
    g, cost = gradientDescent(trainArr, trainRarr, theta, 0.01, 1000)
    print(g)

    # print(result)


    # w = standRegres(np.array(trainSet), np.array(trainRatings))
    # print(type(w))
    # num = len(testSet)
    # cnt = 0.0
    # for i in range(num):
    #     result = np.mat(testSet[i])*w
    #     if abs(result-testRatings[i])<= 1:
    #         cnt += 1.0
    # print(cnt/num)

