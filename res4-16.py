import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/zjm/Playground/logistics-regression/LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
pdData.head()

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]


# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 画图展示sigmoid函数
nums = np.arange(-10, 10, step=1)  # creates a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
# plt.show()


# 在数据中插入一列值为1的列，目的是为了将原来的数值运算转换为矩阵运算。
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


pdData.insert(0, 'Ones', 1)  # 在pdData中添加一列，列名Ones，值为1。
# 设置X（训练数据）和y（目标变量）和theta
orig_data = pdData.values
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
theta = np.zeros([1, 3])  # theta用0填充占位


# 损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))  # 求平均损失


cost(X, y, theta)


# 计算梯度
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # 用0占位，和theta大小一一对应
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


# Gradient descent
# 比较3种不同梯度下降方法

STOP_ITER = 0  # 指定迭代次数
STOP_COST = 1  # 根据损失值目标函数的变化，没有什么变化了就停止
STOP_GRAD = 2  # 根据梯度，如果梯度没什么变化了就停止


def stopCriterion(type, value, threshold):
    # 设定三种不同停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


# 洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break

    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # 对值求解
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    # 以下是辅助函数
    # 根据参数选择梯度下降的方式和停止策略
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += "data - learning rate:{} -".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch({})".format(batchSize)
    name += strDescType + "descent - Stop:"
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_ITER:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    # 画图并展示结果
    print("***{}\nTheta:{} - Iter:{} - Last cost:{:03.2f} - Duration:{:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


n = 100  # 因为样本数就是100，所以n=100时，选择的梯度下降方法是基于所有样本的。
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
plt.show()
