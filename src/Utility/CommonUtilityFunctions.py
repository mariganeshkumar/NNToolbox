import numpy as np


def ValidateDimensionsWithBiasAndOutput(data,weights,biases,output):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1 and output is Dim(L)*NoOfExaples
    if (data.shape[0] != weights.shape[1] or biases.shape[0] != weights.shape[0] or output.shape[0]!=weights.shape[0] or output.shape[1]!=data.shape[1] ):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape, 'biases:', biases.shape, 'output:', output.shape)
        raise ValueError('Incorrect dimmension given to gradient function')


def ValidateDimensionsWithOutput(data,weights,output):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1) and output is Dim(L)*NoOfExaples
    if (data.shape[0] != weights.shape[1]  or output.shape[0]!=weights.shape[0] or output.shape[1]!=data.shape[1] ):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape,  'output:', output.shape)
        raise ValueError('Incorrect dimmension given to gradient function')


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