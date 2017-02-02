import numpy as np
from Utility import CommonUtilityFunctions as cuf


def PureLin(data, weights):
    preActivation=weights*data
    return preActivation


def PureLinWithBias(data,weights,biases,validationRequired=True):
    if validationRequired:
        cuf.ValidateDimensionsWithBias(data,weights,biases)
    DataWithBias,WeightsWithBias=cuf.IntergrateBiasWithWeightsAndData(data,weights,biases)
    return PureLin(DataWithBias,WeightsWithBias)