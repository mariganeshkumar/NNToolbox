import numpy as np
from src.Utility import CommonUtilityFunctions as cuf


def PureLin(data, weights):
    preActivation=np.matmul(weights,data)
    return preActivation


def PureLinWithBias(data,weights,biases,validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBias(data,weights,biases)
    DataWithBias,WeightsWithBias=cuf.IntergrateBiasWithWeightsAndData(data,weights,biases)
    return PureLin(DataWithBias,WeightsWithBias)

def PureLinGradients(activation, gradients, validationRequired=True):
    #todo:Correct the function
    return gradients