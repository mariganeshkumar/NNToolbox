import numpy as np


from src.Activations import Sigmoid
from src.OutputFunctions import SoftMax
from src.LossFunctions import CrossEntropy
from src.Utility import CommonUtilityFunctions as cuf


data=np.array([[0.5,3,0.1]])
weights_l1=np.array([[ 0.19151945 ],[ 0.43772774  ]])#[[ 0.19151945  0.62210877][ 0.43772774  0.78535858]]
biases_l1=np.array([[ 0.62210877],[0.78535858]])
weights_l2=np.array([[ 0.77997581 , 0.27259261  ]
 ,[ 0.80187218  ,0.95813935 ]])#[[ 0.77997581  0.27259261  0.27646426][ 0.80187218  0.95813935  0.87593263]]
biases_l2=np.array([[0.27646426],[0.87593263]])

output=np.array([[0,1,0],[1,0,1]])
layer1Activation=Sigmoid.LogSigmoidWithBias(data,weights_l1,biases_l1)
print('Initial O/p:',SoftMax.SoftMaxWithBias(layer1Activation,weights_l2,biases_l2))
print('Initial O/p Loss:',CrossEntropy.CrossEntropyWithSoftMaxAndBias(layer1Activation,weights_l2,biases_l2,output))
print('Initial Gradients:',CrossEntropy.CrossEntropyWithSoftMaxGradients(layer1Activation,output))
eta=0.1
for i in range(0,50000):
    layer1Activation = Sigmoid.LogSigmoidWithBias(data, weights_l1, biases_l1)
    networkOutput=SoftMax.SoftMaxWithBias(layer1Activation,weights_l2,biases_l2)
    gradientsWRTActivationL2=CrossEntropy.CrossEntropyWithSoftMaxGradients(networkOutput,output)
    gradientsWRTWeightsL2=np.matmul(gradientsWRTActivationL2,np.transpose(cuf.IntergrateBiasAndData(layer1Activation)))
    weights_l2Gradients,biases_l2Gradients=cuf.DisIntergrateBiasFromWeights(gradientsWRTWeightsL2,biasRequired=True)
    gradientsWRTActivationL1 = np.matmul(np.transpose(weights_l2),gradientsWRTActivationL2)
    gradientsWRTActivationL1 = Sigmoid.LogSigmoidGradients(layer1Activation,gradientsWRTActivationL1)
    gradientsWRTWeightsL1 = np.matmul(gradientsWRTActivationL1, np.transpose(cuf.IntergrateBiasAndData(data)))
    weights_l1Gradients, biases_l1Gradients = cuf.DisIntergrateBiasFromWeights(gradientsWRTWeightsL1,biasRequired=True)
    weights_l1=weights_l1-eta*weights_l1Gradients
    weights_l2 = weights_l2 - eta * weights_l2Gradients
    biases_l2=biases_l2-eta*biases_l2Gradients
    biases_l1 = biases_l1 - eta * biases_l1Gradients
    layer1Activation = Sigmoid.LogSigmoidWithBias(data, weights_l1, biases_l1)
    print('log Loss:',CrossEntropy.CrossEntropyWithSoftMaxAndBias(layer1Activation,weights_l2,biases_l2,output))

print('w2g',weights_l2Gradients,'b2g',biases_l2Gradients)
print('w2g',weights_l1Gradients,'b2g',biases_l1Gradients)
layer1Activation=Sigmoid.LogSigmoidWithBias(data,weights_l1,biases_l1)
print('Final O/p:',SoftMax.SoftMaxWithBias(layer1Activation,weights_l2,biases_l2))
