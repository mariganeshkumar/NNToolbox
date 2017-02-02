import numpy as np
from src.Utility import CommonUtilityFunctions as cuf


def SoftMax(data, weights):
    preActivation=np.matmul(weights,data)
    return np.divide(np.exp(preActivation),np.sum(np.exp(preActivation), axis=0))


def SoftMaxWithBias(data,weights,biases,validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBias(data,weights,biases)
    DataWithBias,WeightsWithBias=cuf.IntergrateBiasWithWeightsAndData(data,weights,biases)
    return SoftMax(DataWithBias,WeightsWithBias)