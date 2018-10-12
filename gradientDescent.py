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
    dataSetDF.insert(1, 'Ones', 1)
    cols = dataSetDF.shape[1]
    X = dataSetDF.iloc[:, 1:cols]
    y = dataSetDF.iloc[:, 0]
    # dataSetArr = np.array(dataSetDF)  # np.ndarray()
    # dataSet = dataSetArr.tolist()  # list

    return X,y


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
    # 去掉了前两列以及 Gross的数据 当前返回的为np.array
    X,y = loadData()
    # 来处理训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # trainSet, testSet = train_test_split(dataSet, test_size=0.2)

    X = np.matrix(x_train.values)
    y = np.matrix(y_train.values)
    cols = X.shape[1]
    theta = np.matrix(np.zeros((1, cols)))
    # 1!!!!!!!!
    print(theta)

    print(X.shape, theta.shape, y.shape)
    cost = computeCost(X, y, theta)

    alpha = 0.01
    iters = 5

    # 执行梯度下降算法
    g, cost = gradientDescent(X, y, theta, alpha, iters)

    print(np.shape(x_test))
    # result = sum(x_test[0]*g)
    # print(result)
    # print(y_test[0])

    # print(result)

