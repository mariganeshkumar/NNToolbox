import numpy as np
from src.Activations import Sigmoid
data=np.array([[0.5,2.5]])
weights=np.array([[2]])
biases=np.array([[2]])
output=np.array([[0.2 , 0.9]])
layer1=Sigmoid.LogSigmoidWithBias(data, weights, biases)
outputGradients=output-layer1
weightsGradients,biasesGradients=Sigmoid.LogSigmoidGradientsWithBias(data, weights, biases, outputGradients,biasRequired=True)
print(layer1)
print('wg',weightsGradients,'bg',biasesGradients)
print('present l2 loss',np.linalg.norm(output-layer1))
for i in range(0,500000):
    weights=weights-0.01*weightsGradients
    biases=biases-0.01*biasesGradients
    layer1=Sigmoid.TanSigmoidWithBias(data, weights, biases)
    outputGradients=layer1-output
    weightsGradients,biasesGradients=Sigmoid.TanSigmoidGradientsWithBias(data, weights, biases, outputGradients,biasRequired=True)
    #print(output-layer1)
    #print('wg',weightsGradients,'bg',biasesGradients)
    print('present l2 loss',np.linalg.norm(output-layer1))
print('weights:',weights,'biases',biases)