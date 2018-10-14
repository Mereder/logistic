import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd


def loadData():
    df = pd.read_excel('2014 and 2015 CSM dataset.xlsx')
    dataSetDF = df[['Ratings', 'Budget', 'Screens', 'Sequel']]
    # 'Sentiment', 'Views', 'Likes', 'Dislikes', 'Comments','Budget', 'Screens', 'Sequel'
    dataSetDF = dataSetDF.dropna(how='any')
    cols = dataSetDF.shape[1]
    X = dataSetDF.iloc[:, 1:cols]
    y = dataSetDF.iloc[:, 0]

    X_norm = (X - X.min()) / (X.max() - X.min())
    X_norm.insert(0, 'ones', 1)

    return X_norm,y

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
# 递归下降算法
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
    X,y= loadData()
    # 来处理训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X = np.matrix(x_train.values)
    y = np.matrix(y_train.values)
    X_test = np.matrix(x_test.values)
    y_test = np.matrix(y_test.values)

    w = standRegres(X, y)
    # print(w)
    cnt = 0.0
    y_test = y_test.tolist()
    num = len(y_test[0])
    for i in range(num):
        result = sum(X_test[i,:].dot(w))
        print(result[0,0])
        print(y_test[0][i])
        if abs(result[0,0]-y_test[0][i]) <= 1:
            cnt += 1.0
    print(cnt/num)

