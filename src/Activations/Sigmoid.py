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

#######################################Sigmoid Gradiants###############################################################


def LogSigmoidGradientsWithBias(activations, weights, biases, gradients, validationRequired=True,biasRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBiasActivationAndGradients(activations, weights, biases, gradients)
    weightsWithBias= cuf.IntergrateBiasWithWeights(weights, biases)
    gradients= LogSigmoidGradients(activations, weightsWithBias, gradients, False)
    if biasRequired:
        weightsGradients=gradients[:,0:weights.shape[1]]
        biasesGradients = gradients[:, weights.shape[1]:weights.shape[1]+1]
        return weightsGradients,biasesGradients
    else:
        return gradients


def LogSigmoidGradients(activations, weights, gradients, validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithActivationAndGradients(activations, weights, gradients)
    return np.matmul(np.multiply(np.multiply( gradients, activations), (1 - activations)), np.transpose(data))

def TanSigmoidGradientsWithBias(activations, weights, biases, gradients, validationRequired=True, biasRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBiasActivationAndGradients(activations, weights, biases, gradients)
    weightsWithBias= cuf.IntergrateBiasWithWeights( weights, biases)
    gradients= TanSigmoidGradients(activations, weightsWithBias, gradients, False)
    if biasRequired:
        weightsGradients=gradients[:,0:weights.shape[1]]
        biasesGradients = gradients[:, weights.shape[1]:weights.shape[1]+1]
        return weightsGradients,biasesGradients
    else:
        return gradients


def TanSigmoidGradients(activations, weights, gradients, validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBiasActivationAndGradients(activations, weights, gradients)
    return np.matmul(np.multiply(np.multiply(gradients, (1 + activations)), (1 - activations)), np.transpose(data))


#######################################################################################################################

