import numpy as np
from src.Network import Network
from src.Optimizers import GradientBasedOptimizers as gbo

data=np.array([[0.5,3,0.1]])
weights_l1=np.array([[2],[3]])
biases_l1=np.array([[2],[1]])
weights_l2=np.array([[2,2],[3,3]])
biases_l2=np.array([[2],[1]])

output=np.array([[0,1,0],[1,0,1]])

net = Network.Network([1000,1000],['LogSigmoid','LogSigmoid','TanSigmoid'],'SoftMax','CrossEntropy',1,2)
netOutput,_=net.FeedForward(data)
gbo.BatchGradientDecent(net,data,output,0.001,200)
netOutput,_=net.FeedForward(data)
print(netOutput)