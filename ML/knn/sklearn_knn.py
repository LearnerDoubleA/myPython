import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineSet = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineSet[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('../trainingDigits/%s' % (fileNameStr))
    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('../testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('../testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print('判断：%d 实际：%d' % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print('error rate: %f' % (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()
