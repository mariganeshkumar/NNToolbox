import numpy as np
from src.Network import Network


data=np.array([[0.5,2.5]])
weights=np.array([[2]])
biases=np.array([[2]])
output=np.array([[0.2 , 0.9]])
net = Network.Network([1],['LogSigmoid','LogSigmoid'],'PureLin','SquaredError',1,1)
netOutput,_=net.FeedForward(data)
print(netOutput)
gradients=net.BackProbGradients(data,output)
print(gradients)
net.BatchGradientDecent(data,output,0.1,2000)
netOutput,_=net.FeedForward(data)
print(netOutput)