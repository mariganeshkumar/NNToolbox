import numpy as np



def logSigmoidWithBias(data,weights,biases):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if(data.shape[0]!=weights.shape[1] or biases.shape[0]!=weights.shape[0]):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape,'biases:',biases.shape)
        return
    data = np.append(data,np.ones((1,data.shape[1])),axis=0)
    weights=np.append(weights,biases,axis=1)
    return logSigmoid(data,weights)

def logSigmoid(data,weights):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if (data.shape[0] != weights.shape[1] ):
        print('Error, please check the domension of input matrices')
        print('data:',data.shape,'weights',weights.shape)
        return
    return np.divide(1.0,(1.0+ np.exp(np.matmul(weights,data))))
