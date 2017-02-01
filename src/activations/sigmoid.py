import numpy as np


def LogSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        data,weights=IntergrateBiasWithWeightsAndData(data,weights,biases)
    return LogSigmoid(data, weights, False)

def LogSigmoid(data, weights, validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data,weights)
    return np.divide(1.0,(1.0+ np.exp(np.matmul(weights,data))))

def TanSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        data,weights=IntergrateBiasWithWeightsAndData(data,weights,biases)
    return TanSigmoid(data, weights, False)

def TanSigmoid(data,weights,validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data,weights)
    preActivation=np.matmul(weights,data)
    return np.divide(np.exp(preActivation)-np.exp(-preActivation),np.exp(preActivation)+np.exp(-preActivation))

#######################################################################################################################
def ValidateDimensionsWithBias(data,weights,biases):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1
    if (data.shape[0] != weights.shape[1] or biases.shape[0] != weights.shape[0]):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape, 'biases:', biases.shape)
        raise ValueError('Incorrect dimmension given to sigmoid function')

def ValidateDimensions(data, weights):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1
    if (data.shape[0] != weights.shape[1]):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape)
        raise ValueError('Incorrect dimmension given to sigmoid function')

def IntergrateBiasWithWeightsAndData(data, weights, biases):
    # Input: data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    # Output: data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    data = np.append(data, np.ones((1, data.shape[1])), axis=0)
    weights = np.append(weights, biases, axis=1)
    return data, weights