import numpy as np

from MnistHandler import MnistHandler as mh
from Activations import Sigmoid


ActivationWithBias = {
  'Sigmoid.LogSigmoidWithBias': Sigmoid.LogSigmoidWithBias,
  'Sigmoid.TanSigmoidWithBias': Sigmoid.TanSigmoidWithBias,
}
ActivationGradientsWithBias = {
  'Sigmoid.LogSigmoidWithBias': Sigmoid.LogSigmoidGradientsWithBias,
  'Sigmoid.TanSigmoidWithBias': Sigmoid.TanSigmoidGradientsWithBias,
}

trainData,validData,testData=mh.readMNISTData('../../Data/mnist.pkl.gz')
data=np.array([[0.5,2.5]])
weights=np.array([[2]])
biases=np.array([[2]])
output=np.array([[0.2 , 0.9]])
layer1=ActivationWithBias['Sigmoid.TanSigmoidWithBias'](data, weights, biases)
weightsGradients,biasesGradients=Sigmoid.LogSigmoidGradientsWithBias(data, weights, biases, output)
print(layer1)
print('wg',weightsGradients,'bg',biasesGradients)
print('present l2 loss',np.linalg.norm(output-layer1))
for i in range(0,50000):
    weights=weights-0.01*weightsGradients
    biases=biases-0.01*biasesGradients
    layer1=Sigmoid.TanSigmoidWithBias(data, weights, biases)
    weightsGradients,biasesGradients=Sigmoid.TanSigmoidGradientsWithBias(data, weights, biases, output)
    #print(output-layer1)
    #print('wg',weightsGradients,'bg',biasesGradients)
    print('present l2 loss',np.linalg.norm(output-layer1))
print('weights:',weights,'biases',biases)