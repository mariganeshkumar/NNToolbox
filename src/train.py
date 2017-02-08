import numpy as np
from src.Network import Network
from MnistHandler import MnistHandler as mh
from Activations import Sigmoid
from OutputFunctions import SoftMax
from LossFunctions import CrossEntropy



trainData,validData,testData=mh.readMNISTData('../../Data/mnist.pkl.gz')
data=np.array([[0.5,3,0.1]])
weights_l1=np.array([[2],[3]])
biases_l1=np.array([[2],[1]])
weights_l2=np.array([[2,2],[3,3]])
biases_l2=np.array([[2],[1]])

output=np.array([[0,1,0],[1,0,1]])
net = Network.Network([2, 2],['LogSigmoid','LogSigmoid'],'PureLin','SquaredError',1,2)


net.BatchGradientDecent(data,output,0.8,20000)
netOutput,_=net.FeedForward(data)
print(netOutput)