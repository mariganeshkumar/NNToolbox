import numpy as np

from MnistHandler import MnistHandler as mh
from activations import sigmoid

trainData,validData,testData=mh.readMNISTData('../../Data/mnist.pkl.gz')
data=np.array([[0.5,0.2,5,4.5],[-1,0.8,4,5.1]])
weights=np.array([[0.5,0.2],[1,2],[-0.5,0.2]])
biases=np.array([[0.5],[0.5],[0.5]])
output=np.array([[  3.6e-01 ,  3.1e-01,   2.1e-02  , 2.2e-02],
 [  7.3e-01 ,  9.1e-02,   1.3e-06 ,  2.5e-07],
 [  4.8e-01 ,  3.6e-01 ,  7.6e-01 ,  6.7e-01]])
layer1=sigmoid.LogSigmoidWithBias(data,weights,biases)
gradients=sigmoid.LogSigmoidGradientsWithBias(data,weights,biases,output)
print(output-layer1)
print(gradients)