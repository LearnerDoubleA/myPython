import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def showdata(datingMat, datingLabels):
    font = FontProperties(fname=r'c:/windows/fonts/simsun.ttc', size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('red')
        if i == 2:
            LabelsColors.append('blue')
        if i == 3:
            LabelsColors.append('black')

    axs[0][0].scatter(x=datingMat[:,0], y=datingMat[:,1], color=LabelsColors, s=15,alpha=.5)

    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所小号时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所小号时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=datingMat[:,0], y=datingMat[:,2], color=LabelsColors, s=15, alpha=.5)

    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消耗的冰淇淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingMat[:,1], y=datingMat[:,2], color=LabelsColors, s=15, alpha=.5)

    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消耗的冰淇淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    didntLike = mlines.Line2D([], [], color='black', marker=',', markersize=6, label='不喜欢')
    smallLike = mlines.Line2D([], [], color='blue', marker=',', markersize=6, label='有点喜欢')
    largeLike = mlines.Line2D([], [], color='black', marker=',', markersize=6, label='很喜欢')

    axs[0][0].legend(handle=[didntLike, smallLike, largeLike])
    axs[0][1].legend(handle=[didntLike, smallLike, largeLike])
    axs[1][0].legend(handle=[didntLike, smallLike, largeLike])

    plt.show()

def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVal

def datingClassTest():
    filename = '../datingTestSet2.txt'
    datingMat, datingLabels = file2matrix(filename)
    h = 0.10
    normMat, ranges, minVals = autoNorm(datingMat)
    m = normMat.shape[0]
    numTest = int(m * h)
    error = 0.0
    for i in range(numTest):
        classResult = classify0(normMat[i,:], normMat[numTest:m,:], datingLabels[numTest:m], 4)
        print('分类结果：%d\t实际结果%d' % (classResult, datingLabels[i]))
        if classResult != datingLabels[i]:
            error += 1.0
    print('错误率：%f %%' % (error/float(numTest)*100))

def classifyPerson():
    resultList = ['讨厌', '有点喜欢', '非常喜欢']
    precentTats = float(input('玩视频游戏所耗时间百分比:'))
    ffMiles = float(input('每年获得的飞行常客里程数:'))
    iceCream = float(input('每周消费的冰淇淋公升数:'))

    filename = '../datingTestSet2.txt'
    datingMat, datingLabels = file2matrix(filename)
    normMat, ranges, minmin = autoNorm(datingMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minmin) / ranges
    classResult = classify0(norminArr, normMat, datingLabels, 3)
    print('分类结果：%s' % (resultList[classResult-1]))
'''
if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
'''
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('../trainingDigits/%s' % fileNameStr)
    testFileList = listdir('../testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('预测：%d,实际：%d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('error:%d' % errorCount)
    print('error rate: %f' % (errorCount / float(mTest)))
'''
if __name__ == '__main__':
    handwritingClassTest()
'''
