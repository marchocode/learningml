import numpy as np
import matplotlib.pyplot as plt


# def line regression
def line_regression(w, x, b):
    return w * x + b


def cost_function(y_predict, y_train):
    m = len(y_train)
    sum = 0

    for i in range(m):
        cost = (y_predict[i] - y_train[i]) ** 2
        sum = sum + cost

    return (1 / (2 * m)) * sum


def compute_cost(w, b, x_train, y_train):
    m = len(x_train)
    predict_y = np.zeros(m)

    for i in range(m):
        predict_y[i] = line_regression(w, x_train[i], b)

    return cost_function(predict_y, y_train)


def draw_regression(w, b, x_train, y_train):
    m = len(x_train)
    y_predict = np.zeros(m)

    for i in range(m):
        y_predict[i] = line_regression(w, x_train[i], b)

    plt.scatter(x_train, y_train, c='r', marker='x')
    plt.plot(x_train, y_predict, c='b')
    plt.show()


def get_train_set(start, end, step):
    total = int((end - start) / step)
    zero = np.zeros(total)
    for i in range(total):
        zero[i] = start + (i * step)

    return zero
