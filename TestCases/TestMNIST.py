import numpy as np
from src.Network import Network
from src.MnistHandler import MnistHandler as mh



trainData,valData,testData=mh.readMNISTData('../../Data/mnist.pkl.gz')
trainLabels=trainData[1]
trainData=np.transpose(trainData[0])
trainTargets = np.transpose(np.eye(len(np.unique(trainLabels)))[trainLabels])

valLabels=valData[1]
valData=np.transpose(valData[0])
valTargets = np.transpose(np.eye(len(np.unique(valLabels)))[valLabels])

testLabels=testData[1]
testData=np.transpose(testData[0])
testTargets = np.transpose(np.eye(len(np.unique(testLabels)))[testLabels])
net = Network.Network([200, 200 ],['LogSigmoid','LogSigmoid'],'SoftMax','CrossEntropy',784,10,'/tmp')
net.MiniBatchGradientDecentWithMomentum(trainData,trainTargets,20000,200,eta=0.8,gamma=0.2,
                                        valData=valData,valTargets=valTargets,testData=testData,testTargets=testTargets)
print (valData.shape)
