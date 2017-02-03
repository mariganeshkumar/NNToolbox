import numpy as np
from src.Network import Network

data=np.array([[0.5,3,0.1]])
weights_l1=np.array([[2],[3]])
biases_l1=np.array([[2],[1]])
weights_l2=np.array([[2,2],[3,3]])
biases_l2=np.array([[2],[1]])

output=np.array([[0,1,0],[1,0,1]])

net = Network.Network([5,5],['LogSigmoid','LogSigmoid'],'SoftMax','CrossEntropy',1,2)
netOutput,_=net.FeedForward(data)
net.BatchGradientDecent(data,output,0.9,200000)
netOutput,_=net.FeedForward(data)
print(netOutput)