# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader

#  sigmoid函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#  sigmoid函数的导数，激活函数
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

#  神经网络的类
class Network(object):
    def __init__(self, sizes):
        self.numOfLayers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]  # 随机初始化权重
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]    # 随机初始化偏置

    #  定义feedforward函数
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):  # 注意zip函数的具体用法
            a = sigmoid(np.dot(w, a) + b)
        return a

    #  定义随机梯度下降函数，赋予神经网络学习的能力：
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        len_test = len(test_data)
        n = len(training_data)  # 训练数据大小
        #  迭代过程
        for j in range(epochs):
            random.shuffle(training_data)
            #  mini_batches是列表中放切割之后的列表
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            #  每个mini_batch都更新一次,重复完整个数据集
            for mini_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                eta = learning_rate / len(mini_batch)
                #  mini_batch中的一个实例调用梯度下降得到各个参数的偏导
                for x, y in mini_batch:
                    #  从一个实例得到的梯度
                    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                # 每一个mini_batch更新一下参数
                self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]
                self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]

            reslut = self.evaluate(test_data)
            print("Epoch " + str(j) + '--> ' + 'accuracy:' + str(reslut[0]/len_test) +
                  '  ' + 'cost:'+str(reslut[1]))
            accuracy.append(reslut[0]/len_test)
            cost.append(reslut[1])

    # 反向传播
    def backprop(self, x, y):
        #  存储C对于各个参数的偏导
        #  格式和self.biases和self.weights是一模一样的
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #  前向过程
        activation = x
        activations = [x]  # 存储所有的激活值,一层一层的形式
        zs = []  # 存储所有的中间值(weighted sum)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #  反向过程
        #  计算输出层error
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #  非输出层
        for l in range(2, self.numOfLayers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    #  输出层cost函数对于a的导数
    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        total = 0
        for (x, y) in test_result:
            if x != y:
                total = total + (x - y) ** 2
        error = total**0.5 / len(test_data)
        return sum(int(i == j) for (i, j) in test_result), error


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    cost = []  # 记录损失
    accuracy = []  # 记录准确率
    epoch = list(range(30))
    net = Network([784, 30, 10])    # 输入层、隐藏层、输出层的结点数量
    net.SGD(training_data, len(epoch), 10, 3.0, test_data)
    #  figure创建一个绘图对象 figsize 图片大小
    new_ticks = np.linspace(0, len(epoch), len(epoch)+1)
    plt.xticks(new_ticks)
    plt.plot(epoch, accuracy)
    plt.show()

    new_ticks = np.linspace(0, len(epoch), len(epoch) + 1)
    plt.xticks(new_ticks)
    plt.plot(epoch, cost)
    plt.show()
