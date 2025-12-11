import numpy as np
import copy
import math
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


def compute_cost_multi(w, b, X_train, y_train):
    """
    计算多特性输入下的损失函数
    :param w:
    :param b:
    :param X_train:
    :param y_train:
    :return:
    """

    m = X_train.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X_train[i], w) + b
        cost = cost + (y_train[i] - f_wb_i) ** 2

    cost = cost / (2 * m)
    return cost


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


def gradient_function_multi(w, b, x_train, y_train):
    """
    梯度下降，支持多个参数
    :param w:
    :param b:
    :param x_train:
    :param y_train:
    :return:
    """
    m, n = x_train.shape

    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        err = (np.dot(x_train[i], w) + b) - y_train[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x_train[i, j]

        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db


def gradient_descent_multi(w_init, b_init, x_train, y_train, alpha, num_iterations):
    """
    运行梯度下降,获得最佳w和b
    :param w_init:
    :param b_init:
    :param x_train:
    :param y_train:
    :param alpha:
    :param num_iterations:
    :return:
    """
    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init)
    cost_history = []

    for i in range(num_iterations):

        dj_dw, dj_db = gradient_function_multi(w, b, x_train, y_train)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:  # prevent resource exhaustion
            cost = compute_cost_multi(w, b, x_train, y_train)
            cost_history.append(cost)

        if i % math.ceil(num_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost_history[-1]:8.2f}")

    return w, b
