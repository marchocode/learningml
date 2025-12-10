import numpy as np
import matplotlib.pyplot as plt


# def line regression
def line_regression(w, x, b):
    """
    线性回归函数 y=wx+b
    :param w:
    :param x:
    :param b:
    :return: y
    """
    return w * x + b


def cost_function(y_predict, y_train):
    """
    损失函数
    :param y_predict: 预测值
    :param y_train: 答案
    :return:
    """
    m = len(y_train)
    sum = 0

    for i in range(m):
        cost = (y_predict[i] - y_train[i]) ** 2
        sum = sum + cost

    return (1 / (2 * m)) * sum


def compute_cost(w, b, x_train, y_train):
    """
    使用线性回归计算指定参数w和b的总体损失
    :param w:
    :param b:
    :param x_train:
    :param y_train:
    :return:
    """
    m = len(x_train)
    predict_y = np.zeros(m)

    for i in range(m):
        predict_y[i] = line_regression(w, x_train[i], b)

    return cost_function(predict_y, y_train)


def draw_regression(w, b, x_train, y_train):
    """
    画出一条w和b参数构成的拟合线
    :param w:
    :param b:
    :param x_train:
    :param y_train:
    :return:
    """
    m = len(x_train)
    y_predict = np.zeros(m)

    for i in range(m):
        y_predict[i] = line_regression(w, x_train[i], b)

    plt.scatter(x_train, y_train, c='r', marker='x')
    plt.plot(x_train, y_predict, c='b')
    plt.show()


def get_train_set(start, end, step):
    """
    获得测试训练集
    :param start:
    :param end:
    :param step:
    :return:
    """
    total = int((end - start) / step)
    zero = np.zeros(total)
    for i in range(total):
        zero[i] = start + (i * step)

    return zero


def gradient_function(w, b, x_train, y_train):
    """
    梯度下降
    :param w:
    :param b:
    :param x_train:
    :param y_train:
    :return:
    """
    m = len(x_train)

    dj_w = 0
    dj_b = 0

    for i in range(m):
        y_predict = line_regression(w, x_train[i], b)
        dj_wi = (y_predict - y_train[i]) * x_train[i]
        dj_bi = y_predict - y_train[i]

        dj_w = dj_w + dj_wi
        dj_b = dj_b + dj_bi

    dj_w = dj_w / m
    dj_b = dj_b / m

    return dj_w, dj_b