from PIL import Image
import numpy as np
import os
import re


def read_pgm(filename, byteorder='>'):
    """
    读取pgm格式图片文件,参照网上代码，自己并不会写
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def read_bmp(filename):
    """
    读取bmp格式图片文件
    """
    image = Image.open(filename)
    data = image.getdata()
    temp = []
    for i in data:
        temp.append(i)
    return temp  # 返回一维行向量


def load_path(path):
    """
    把训练集和测试集中的数据路径进行处理，得到相应的列表
    """
    datasetsPath = os.path.join(path, 'Datasets')

    # 定义att和yale的训练集、测试集的文件目录的列表结构
    att_Train_Dir = []
    att_Test_Dir = []
    att_Train_Num = []
    att_Test_Num = []

    yale_Train_Dir = []
    yale_Test_Dir = []
    yale_Train_Num = []
    yale_Test_Num = []

    # 加载att数据集
    att_Path = os.path.join(datasetsPath, 'att')  # att数据集的绝对路径
    att_Dirs = os.listdir(att_Path)
    # ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17']
    for i in range(len(att_Dirs)):
        test_Dir = os.path.join(os.path.join(att_Path, att_Dirs[i]), 'test')  # 测试集路径 E:\Python\PCA\Datasets\att\s1\test
        train_Dir = os.path.join(os.path.join(att_Path, att_Dirs[i]), 'train')  # 训练集路径

        testPgms = os.listdir(os.path.join(os.path.join(att_Path, att_Dirs[i]),
                                           'test'))  # 测试图片文件名称列表 ['3.pgm', '5.pgm', '6.pgm', '8.pgm']
        trainPgms = os.listdir(os.path.join(os.path.join(att_Path, att_Dirs[i]), 'train'))  # 训练图片文件名称列表
        for j in testPgms:
            att_Test_Dir.append(os.path.join(test_Dir, j))
            att_Test_Num.append(i + 1)
        for j in trainPgms:
            att_Train_Dir.append(os.path.join(train_Dir, j))
            att_Train_Num.append(i + 1)

    # 加载yale数据集
    yale_Path = os.path.join(datasetsPath, 'yale')  # yale数据集的绝对路径
    yale_Dirs = os.listdir(yale_Path)
    # ['s1', 's10', 's11', 's12', 's13']
    for i in range(len(yale_Dirs)):  # 加载yale数据集
        test_Dir = os.path.join(os.path.join(yale_Path, yale_Dirs[i]), 'test')  # 测试集路径
        train_Dir = os.path.join(os.path.join(yale_Path, yale_Dirs[i]), 'train')  # 训练集路径
        testBmps = os.listdir(test_Dir)  # 测试集图片
        trainBmps = os.listdir(train_Dir)  # 训练集图片
        for j in testBmps:
            yale_Test_Dir.append(os.path.join(test_Dir, j))
            yale_Test_Num.append(i + 1)
        for j in trainBmps:
            yale_Train_Dir.append(os.path.join(train_Dir, j))
            yale_Train_Num.append(i + 1)

    return att_Train_Dir, att_Test_Dir, att_Train_Num, att_Test_Num, \
           yale_Train_Dir, yale_Test_Dir, yale_Train_Num, yale_Test_Num


def load_data(path):
    """
    利用上面的三个函数，进行图片数据的读取
    """
    curdir = path

    att_Train_Dir, att_Test_Dir, att_Train_Num, att_Test_Num, \
    yale_Train_Dir, yale_Test_Dir, yale_Train_Num, yale_Test_Num = load_path(curdir)

    attTrainData = []  # att中的训练集数据
    attTestData = []  # att中的测试集数据

    yaleTrainData = []  # yale中的训练集数据
    yaleTestData = []  # yale中的测试集数据

    for i in att_Train_Dir:  # 读取attTrainData
        matrix = np.array(read_pgm(i, byteorder='<'))
        # print(matrix)
        data = matrix.reshape(1, matrix.shape[0] * matrix.shape[1])
        attTrainData.append(data.tolist()[0])

    for i in att_Test_Dir:  # 读取attTestData
        matrix = np.array(read_pgm(i, byteorder='<'))

        #  重塑数组，变成一个行向量
        total = matrix.shape[0] * matrix.shape[1]  # 行向量的列数
        data = matrix.reshape(1, total)

        #  转成列表，当矩阵是1 * n维的时候，经常会有tolist()[0]
        attTestData.append(data.tolist()[0])

    for i in yale_Train_Dir:
        data = read_bmp(i)
        yaleTrainData.append(data)

    for i in yale_Test_Dir:
        data = read_bmp(i)
        yaleTestData.append(data)

    return np.array(attTrainData), np.array(attTestData), np.array(att_Train_Num), \
           np.array(att_Test_Num), np.array(yaleTrainData), np.array(yaleTestData), \
           np.array(yale_Train_Num), np.array(yale_Test_Num)


def PCA(data, n):
    """
    PCA算法实现的基本过程
    参考网上代码：https://blog.csdn.net/on2way/article/details/47059203
    """
    # （1）零均值化
    matrix = np.mat(data)
    dataMat = np.float32(matrix)  # 转为浮点型matrix
    rows, cols = dataMat.shape  # 取大小
    data_mean = np.mean(dataMat, 0)  # 对列求均值
    data_mean_all = np.tile(data_mean, (rows, 1))
    Temp = dataMat - data_mean_all  # 减去均值

    # （2）简单的方法求协方差矩阵
    T1 = Temp * Temp.T  # 使用矩阵计算，所以前面要进行mat转化

    # （3）求特征值、特征矩阵
    D, V = np.linalg.eig(T1)  # 特征值与特征矩阵

    # （3）保留主要的成分[即保留值比较大的前n个特征]
    V1 = V[:, 0:n]    # 取前n个特征向量
    V1 = Temp.T * V1  # 取前n个特征向量
    for i in range(n):  # 特征向量归一化
        L = np.linalg.norm(V1[:, i])
        V1[:, i] = V1[:, i] / L

    data_new = Temp * V1  # 降维后的数据,V1是最终的特征
    return data_new, data_mean, V1


def main():
    """
    参照代码：https://blog.csdn.net/on2way/article/details/47059203
    说实话，不是很懂
    """

    curdir = os.path.split(os.path.realpath(__file__))[0]  # 当前文件所在路径

    att_Train_Data, att_Test_Data, att_Train_Num, att_Test_Num, yale_Train_Data, \
    yale_Test_Data, yale_Train_Num, yale_Test_Num = load_data(curdir)  # 得到数据集

    print("——————————————————————att数据集——————————————————————")
    for n in range(10, 41, 2):  # 维度的变化范围从10~40
        # 利用PCA算法进行训练
        low_data_train, data_mean, V_n = PCA(att_Train_Data, n)
        temp_face = att_Test_Data - data_mean
        low_data_test = temp_face * V_n  # 得到测试脸在特征向量下的数据
        low_data_test = np.array(low_data_test)  # matrix转array
        # print(low_data_test)
        low_data_train = np.array(low_data_train)

        # 计算准确度
        true_num = 0

        # print(att_Test_Data.shape[0])     第一维的长度，即样本数量

        for i in range(att_Test_Data.shape[0]):  # 每一张测试脸对所有训练脸求欧式距离
            testFace = low_data_test[i]  # 取数组的第i行
            diffMat = low_data_train - testFace  # 训练数据与测试脸之间距离
            sqDiffMat = diffMat ** 2    # 对数组中的所有元素取平方
            sqDistances = sqDiffMat.sum(axis=1)  # 按行求和,得到一个列表

            sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
            indexMin = sortedDistIndicies[0]  # 距离最近的索引
            # print(indexMin)
            # print('---')
            if att_Train_Num[indexMin] == att_Test_Num[i]:  # 如果标记相同，则判断正确
                true_num += 1
            else:
                pass

        accuracy = float(true_num) / att_Test_Data.shape[0]
        print('                 %d维---->准确率: %.2f%%' % (n, accuracy * 100))



    print("——————————————————————yale数据集——————————————————————")
    for n in range(10, 21, 2):
        # 利用PCA算法进行训练
        low_data_train, data_mean, V_n = PCA(yale_Train_Data, n)
        temp_face = yale_Test_Data - data_mean
        low_data_test = temp_face * V_n  # 得到测试脸在特征向量下的数据
        low_data_test = np.array(low_data_test)  # matrix转array
        low_data_train = np.array(low_data_train)

        # 计算准确度
        true_num = 0
        for i in range(yale_Test_Data.shape[0]):  # 每一张测试脸对所有训练脸求欧式距离
            testFace = low_data_test[i]
            diffMat = low_data_train - testFace  # 训练数据与测试脸之间距离
            sqDiffMat = diffMat ** 2    # 对数组中的所有元素取平方
            sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
            sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
            indexMin = sortedDistIndicies[0]  # 距离最近的索引
            if yale_Train_Num[indexMin] == yale_Test_Num[i]:
                true_num += 1
            else:
                pass

        accuracy = float(true_num) / yale_Test_Data.shape[0]
        print('                 %d维---->准确率: %.2f%%' % (n, accuracy * 100))


if __name__ == '__main__':
    main()
