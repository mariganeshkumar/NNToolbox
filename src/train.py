import numpy as np

from MnistHandler import MnistHandler as mh
from activations import sigmoid

trainData,validData,testData=mh.readMNISTData('../../Data/mnist.pkl.gz')
data=np.array([[0.5,2.5]])
weights=np.array([[2]])
biases=np.array([[2]])
output=np.array([[0.2 , 0.9]])
layer1=sigmoid.LogSigmoidWithBias(data,weights,biases)
weightsGradients,biasesGradients=sigmoid.LogSigmoidGradientsWithBias(data,weights,biases,output)
print(layer1)
print('wg',weightsGradients,'bg',biasesGradients)
print('present l2 loss',np.linalg.norm(output-layer1))
for i in range(0,500000):
    weights=weights-0.01*weightsGradients
    biases=biases-0.01*biasesGradients
    layer1=sigmoid.LogSigmoidWithBias(data,weights,biases)
    weightsGradients,biasesGradients=sigmoid.LogSigmoidGradientsWithBias(data,weights,biases,output)
    #print(output-layer1)
    #print('wg',weightsGradients,'bg',biasesGradients)
    print('present l2 loss',np.linalg.norm(output-layer1))
print('weights:',weights,'biases',biases)