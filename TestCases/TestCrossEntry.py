import numpy as np


from src.Activations import Sigmoid
from src.OutputFunctions import SoftMax
from src.LossFunctions import CrossEntropy


data=np.array([[0.5,2.5,0.1]])
weights_l1=np.array([[2],[3]])
biases_l1=np.array([[2],[1]])
weights_l2=np.array([[2,2],[3,3]])
biases_l2=np.array([[2],[1]])

output=np.array([[0,1,0],[1,0,1]])
layer1Activation=Sigmoid.LogSigmoidWithBias(data,weights_l1,biases_l1)
print('Initial O/p:',SoftMax.SoftMaxWithBias(layer1Activation,weights_l2,biases_l2))
print('Initial O/p Loss:',CrossEntropy.CrossEntropyWithSoftMaxAndBias(layer1Activation,weights_l2,biases_l2,output))
print('Initial Gradients:',CrossEntropy.CrossEntropyWithSoftMaxGradients(layer1Activation,weights_l2,biases_l2,output))
eta=2.4
for i in range(0,50000):
    layer1Activation = Sigmoid.LogSigmoidWithBias(data, weights_l1, biases_l1)
    Layer_l2Gradients=CrossEntropy.CrossEntropyWithSoftMaxGradients(layer1Activation,weights_l2,biases_l2,output)
    weights_l2Gradients,biases_l2Gradients=Sigmoid.LogSigmoidGradientsWithBias(layer1Activation,weights_l2,
                                                                               biases_l2,Layer_l2Gradients,
                                                                               biasRequired=True)
    weights_l1Gradients, biases_l1Gradients = Sigmoid.LogSigmoidGradientsWithBias(data, weights_l1,
                                                                                  biases_l1,
                                                                                  np.append(weights_l2Gradients,
                                                                                            biases_l2Gradients,axis=1),
                                                                                  biasRequired=True)
    weights_l1=weights_l1-eta*weights_l1Gradients
    weights_l2 = weights_l2 - eta * weights_l2Gradients
    biases_l2=biases_l2-eta*biases_l2Gradients
    biases_l1 = biases_l1 - eta * biases_l1Gradients
    print('log Loss:',CrossEntropy.CrossEntropyWithSoftMaxAndBias(layer1Activation,weights_l2,biases_l2,output))

print('w2g',weights_l2Gradients,'b2g',biases_l2Gradients)
print('w2g',weights_l1Gradients,'b2g',biases_l1Gradients)
layer1Activation=Sigmoid.LogSigmoidWithBias(data,weights_l1,biases_l1)
print('Final O/p:',SoftMax.SoftMaxWithBias(layer1Activation,weights_l2,biases_l2))
