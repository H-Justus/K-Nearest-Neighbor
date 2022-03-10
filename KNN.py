# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 15:01
# @Author  : Justus
# @FileName: KNN.py
# @Software: PyCharm

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# 清理变量
try:
    del X_train, y_train
    del X_test, y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    # idxs为对应标签的下标
    idxs = np.flatnonzero(y_train == y)
    # 从idxs中随机取samples_per_class个不重复的元素
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        # 7*10输出
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# 训练集取5000张，测试集取500张
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# 图像一维化(5000, 32, 32, 3)(500, 32, 32, 3)to(5000, 3072) (500, 3072)
print(X_train.shape, X_test.shape)
print("Reshape the image data into rows:")
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# 创建一个KNN分类器(KNN分类器只记录数据不训练)
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 测试分类器
dists = classifier.compute_distances_two_loops(X_test)
print(type(dists), dists.shape)

# 可视化距离矩阵:每一行都是一个单独的测试示例与训练例子的距离(最小值为黑最大值为白，黑色近白色远)
# plt.imshow(dists, interpolation='none')
# plt.show()

# 计算准确率
y_test_pred = classifier.predict_labels(dists, k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print('Got %d / %d correct => accuracy: %f' % (int(num_correct), num_test, accuracy))

# 查看距离是否计算正确
dists_one = classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('One loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('The distance matrices are the same')
else:
    print('The distance matrices are different')

dists_two = classifier.compute_distances_two_loops(X_test)
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Two loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('The distance matrices are the same')
else:
    print('The distance matrices are different')

dists_no = classifier.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists_no, ord='fro')
print('No loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('The distance matrices are the same')
else:
    print('The distance matrices are different')


def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)
no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# 交叉验证
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

# 将训练数据分解，X_train_folds和y_train_folds都应该是长度为num_folds的列表
# 其中y_train_folds [i]是X_train_folds [i]中的点的标签向量
print("X_train, y_train shape", X_train.shape, y_train.shape)
X_train_folds = np.array(np.split(X_train, num_folds))
y_train_folds = np.array(np.split(y_train, num_folds))
print("X_train_folds, y_train_folds shape", X_train_folds.shape, y_train_folds.shape)
# k_to_accuracies [k]应该是一个长度为num_folds的字典，给出了我们在使用k值时发现的不同精度值。
k_to_accuracies = {}
# 执行k-fold交叉验证以找到k的最佳值。
# 对于k的每个可能值，运行k-nearest-neighbor算法num_folds次，
# 在每种情况下，你使用所有的折叠作为训练数据，最后一个折叠作为验证集。
# 将所有折叠的精度和k的所有值存储在k to accuracies字典中。
for k in k_choices:
    curr_acc = []
    for i in np.arange(num_folds):
        indx = np.array([j for j in range(num_folds) if j != i])
        X_test_n = X_train_folds[i]
        y_test_n = y_train_folds[i]
        X_train_n = np.concatenate(X_train_folds[indx], axis=0)
        y_train_n = np.concatenate(y_train_folds[indx], axis=None)
        # 进行训练
        classifier = KNearestNeighbor()
        classifier.train(X_train_n, y_train_n)
        dists = classifier.compute_distances_no_loops(X_test_n)
        y_test_n_pred = classifier.predict_labels(dists, k)
        num_correct = np.sum(y_test_n_pred == y_test_n)
        accuracy = float(num_correct) / len(y_test_n)
        curr_acc.append(accuracy)
    k_to_accuracies[k] = curr_acc

# 打印出准确度
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# 绘制趋势图
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# 观察得到最佳k值
best_k = 10
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)
# 计算和显示精度
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (float(num_correct), num_test, accuracy))
