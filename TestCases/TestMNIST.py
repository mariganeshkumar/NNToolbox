import numpy as np
from src.Network import Network
from src.MnistHandler import MnistHandler as mh
from src.Optimizers import GradientBasedOptimizers as gbo

from sklearn.decomposition.pca import PCA
# pca = PCA(n_components=392, whiten=True)




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

# pca.fit(np.transpose(np.r_['-1',trainData,valData,testData]))
# trainData=np.transpose(pca.transform(np.transpose(trainData)))
# valData=np.transpose(pca.transform(np.transpose(valData)))
# testData=np.transpose(pca.transform(np.transpose(testData)))
net = Network.Network([200,200,200],['TanSigmoid','TanSigmoid','TanSigmoid','ReLU','ReLU','TanSigmoid','TanSigmoid'],'SoftMax','CrossEntropy',trainData.shape[0],10,'/tmp')
net=gbo.PreTrainNetwork(net,trainData,20000,200,0.01,0.05,valData=valData,annel=True,regularization=True,lamda=0.1)
net=gbo.NestrovAccelaratedGradientDecent(net,trainData,trainTargets,200000,200,eta=0.2,gamma=0.05,
                                        valData=valData,valTargets=valTargets,testData=testData,testTargets=testTargets,
                                        annel=True,regularization=True,lamda=0.1)
print (valData.shape)
