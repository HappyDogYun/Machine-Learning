import pickle
import gzip
import numpy as np

def load_data():
    """
    解压数据集,然后从数据集中把数据取出来
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    之前的load_data返回的格式虽然很漂亮,但是并不是非常适合我们这里计划的
    神经网络的结构,因此我们在load_data的基础上面使用load_data_wrapper函数
    来进行一点点适当的数据集变换,使得数据集更加适合我们的神经网络训练.
    """
    tr_d, va_d, te_d = load_data()
    #  训练集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data =[(x, y)for x, y in zip(training_inputs, training_results)]

    #  验证集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = [(x, y)for x, y in zip(validation_inputs, va_d[1])]

    #  测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = [(x, y)for x, y in zip(test_inputs, te_d[1])]

    return training_data, validation_data, test_data

def vectorized_result(j):
    #  形状为10行1列
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
