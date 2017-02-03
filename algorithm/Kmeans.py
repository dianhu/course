#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 16-9-19 下午5:15
# @Author  : H.C.Y
# @File    : Kmeans.py
# @Descripe: 自己实现的k_means算法
import numpy as np
import matplotlib.pyplot as plt

# 两个向量间的欧式距离
def ed_distance(v1, v2):
    return np.sqrt(np.sum(np.power(v2 - v1, 2)))


# 根据k，随机初始化中心点
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


def k_means(dataSet, k):
    numSamples = dataSet.shape[0]  # 样本数量
    clusterAssment = np.mat(np.zeros((numSamples, 2)))#存储　(样本所属cluster，样本距离类中心距离)
    clusterChanged = True

    # step:1 初始化中心点
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):  # 循环所有样本去和中心点计算距离
            minDis = 10000000.0
            minInd = 0
            # step2:找到距离样本最近的中心点
            for j in range(k):
                dis = ed_distance(centroids[j, :], dataSet[i, :])
                if dis < minDis:
                    minDis = dis
                    minInd = j

            # step3:更新样本的类别
            if clusterAssment[i, 0] != minInd:
                clusterChanged = True
                clusterAssment[i, :] = minInd, minDis ** 2

        # setp4:更新中心点
        for j in range(k):
            #取得所属cluster的样本求平均值更新中心点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusteAassment):

    numSamples, dim = dataSet.shape

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=10)

    plt.show()


if __name__ == "__main__":
    # step 1: load data
    print "step 1: load data..."
    dataSet = []
    fileIn = open('testData.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    # step 2: clustering...
    print "step 2: clustering..."
    dataSet = np.mat(dataSet)
    k = 4
    centroids, clusterAssment = k_means(dataSet, k)
    # step 3: show the result
    print "step 3: show the result..."
    showCluster(dataSet, k, centroids, clusterAssment)
