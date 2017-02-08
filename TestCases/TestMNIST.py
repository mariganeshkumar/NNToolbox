import numpy as np
from src.Network import Network
from src.MnistHandler import MnistHandler as mh
from src.Optimizers import GradientBasedOptimizers as gbo



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
net = Network.Network([200, 200],['TanSigmoid','TanSigmoid','TanSigmoid','TanSigmoid','TanSigmoid','TanSigmoid','TanSigmoid'],'SoftMax','CrossEntropy',784,10,'/tmp')
net=gbo.NestrovAccelaratedGradientDecent(net,trainData,trainTargets,20000,200,eta=0.5,gamma=0.1,
                                        valData=valData,valTargets=valTargets,testData=testData,testTargets=testTargets,
                                        annel=True)
print (valData.shape)
