import numpy as np

from Utility.CommonUtilityFunctions import ValidateDimensionsWithBiasAndOutput, ValidateDimensionsWithOutput, \
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


def LogSigmoidGradientsWithBias(data,weights,biases,output,validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithBiasAndOutput(data, weights, biases, output)
    data,weightsWithBias= IntergrateBiasWithWeightsAndData(data, weights, biases)
    gradients= LogSigmoidGradients(data,weightsWithBias,output,False)
    weightsGradients=gradients[:,0:weights.shape[1]]
    biasesGradients = gradients[:, weights.shape[1]:weights.shape[1]+1]
    return weightsGradients,biasesGradients


def LogSigmoidGradients(data, weights, output, validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithOutput(data, weights, output)
    activations=LogSigmoid(data,weights,False)
    return np.matmul(np.multiply(np.multiply((activations - output),activations),(1-activations)),np.transpose(data))

def TanSigmoidGradientsWithBias(data,weights,biases,output,validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithBiasAndOutput(data, weights, biases, output)
    data,weightsWithBias= IntergrateBiasWithWeightsAndData(data, weights, biases)
    gradients= TanSigmoidGradients(data,weightsWithBias,output,False)
    weightsGradients=gradients[:,0:weights.shape[1]]
    biasesGradients = gradients[:, weights.shape[1]:weights.shape[1]+1]
    return weightsGradients,biasesGradients


def TanSigmoidGradients(data, weights, output, validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithOutput(data, weights, output)
    activations=TanSigmoid(data,weights,False)
    return np.matmul(np.multiply(np.multiply((activations - output),(1+activations)),(1-activations)),np.transpose(data))


#######################################################################################################################

