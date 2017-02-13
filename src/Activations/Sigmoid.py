import numpy as np

from src.Utility import CommonUtilityFunctions as cuf

from src.Utility.CommonUtilityFunctions import ValidateDimensionsWithBiasAndOutput, ValidateDimensionsWithOutput, \
    ValidateDimensionsWithBias, ValidateDimensions, IntergrateBiasWithWeightsAndData


def LogSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        ValidateDimensionsWithBias(data, weights, biases)
    data,weights= IntergrateBiasWithWeightsAndData(data, weights, biases)
    return LogSigmoid(data, weights, False)

def LogSigmoid(data, weights, validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data, weights)
    return np.divide(1.0,(1.0+ np.exp(-1*np.matmul(weights,data))))

def TanSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        ValidateDimensionsWithBias(data, weights, biases)
    data,weights= IntergrateBiasWithWeightsAndData(data, weights, biases)
    return TanSigmoid(data, weights, False)

def TanSigmoid(data,weights,validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data, weights)
    preActivation=np.matmul(weights,data)
    return np.divide(np.exp(preActivation)-np.exp(-1*preActivation),np.exp(preActivation)+np.exp(-1*preActivation))

def ReLU(data,weights,validationRequired=True):
    if validationRequired:
        ValidateDimensions(data, weights)
    preActivation=np.matmul(weights,data)
    return np.maximum(preActivation,0)


#######################################Sigmoid Gradiants###############################################################




def LogSigmoidGradients(activations, gradients, validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithActivationAndGradients(activations, gradients)
    return np.multiply(np.multiply( gradients, activations), (1 - activations))



def TanSigmoidGradients(activations, gradients, validationRequired=True):
    #todo: remove weights
    if validationRequired:
        cuf.ValidateDimensionsWithActivationAndGradients(activations, gradients)
    return np.multiply(np.multiply(gradients, (1 + activations)), (1 - activations))

def ReLUGradients(activations, gradients, validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithActivationAndGradients(activations, gradients)
        reluGradient=activations
        reluGradient[np.greater(activations,0)]=1
    return np.multiply(gradients, reluGradient)

#######################################################################################################################

