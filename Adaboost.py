import xgboost as xgb
from numpy import *
#boosting是通过集中关注已有分类器错分的那些数据来获得新的分类器，boosting分类的结果是基于所有分类器的加权求和结果的，bagging中的分类器权重是相等的
#
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the lstm-ptb-data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                print ("predictedVals",predictedVals.T,"errArr",errArr.T)
                weightedError = D.T*errArr  #calc total error multiplied by D
                print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
def adaBoostTrain(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "error",error
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        print ("alpha",alpha)
        weakClassArr.append(bestStump)#store Stump Params in Array
        print ("classEst",classEst)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon)) #Calc New D, element-wise
        D = D/D.sum()
        print ("D",D)
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print ("aggClassEst",aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print (errorRate)
        if errorRate == 0.0: break
    return weakClassArr
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    #该函数的输入是由一个或多个待分类样例datToClass以及多个弱分类器组成的数组classfierArr
    #首先将datToClass转换成一个矩阵，并得到待分类样例的个数m
    dataMatrix = mat(datToClass)
    #do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
   #创建一个0列向量
    aggClassEst = mat(zeros((m,1)))
    #遍历calssifierArr中所有弱分类器，并给予stumpClassify对每个分类器得到一个类别的估计值。
    for i in range(len(classifierArr)):
        #stumpClassify函数中，我们在所有可能的决策树上进行迭代得到具有最小加权错误率的单层决策树
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #call stump classify
        #输出的类别估计值乘上该单层决策树的alpha权重然后累加
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
# datMat,classLabels=loadSimpData()
# D=mat(ones((5,1))/5)
# print(buildStump(datMat,classLabels,D))
#
# classifierArray=adaBoostTrainDS(datMat,classLabels,9)
# print(classifierArray)

# datArr,labelArr=loadSimpData()
# classifierArr=adaBoostTrainDS(datArr,labelArr,30)
#
# print(adaClassify([[5,5],[0,0]],classifierArr))

datArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArr=adaBoostTrainDS(datArr,labelArr,10)
print(classifierArr)