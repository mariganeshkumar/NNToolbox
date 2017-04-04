import numpy as np

from src.Activations import Sigmoid
from src.LossFunctions import CrossEntropy,SquaredError
from src.OutputFunctions import SoftMax,PureLin
from src.Utility import CommonUtilityFunctions as cuf


class Network:


    def __init__(self, hiddenLayers, activationFunctionNames, outputFunctionName, lossFunctionName, noOfInputs,
                 noOfOutputs, logDir=None):
        self.hiddenLayers = hiddenLayers
        self.noOfLayers = len(hiddenLayers)
        self.activationFunctionNames = activationFunctionNames
        self.outputFunctionName = outputFunctionName
        self.lossFunctionName=lossFunctionName
        self.noOfInputs = noOfInputs
        self.noOfOutputs = noOfOutputs
        np.random.seed(seed=1234)
        self.weights=[]
        self.weights.append(0.1*np.random.randn(hiddenLayers[0], noOfInputs + 1))
        self.logDir=logDir
        for i in range(1, len(hiddenLayers)):
            self.weights.append(0.1*np.random.randn(hiddenLayers[i], hiddenLayers[i - 1] + 1))
        self.weights.append(0.1*np.random.randn(noOfOutputs, hiddenLayers[-1]+1))

        self.Activation = {
            'LogSigmoid': Sigmoid.LogSigmoid,
            'TanSigmoid': Sigmoid.TanSigmoid,
            'ReLU': Sigmoid.ReLU,
        }
        self.ActivationGradients = {
            'LogSigmoid': Sigmoid.LogSigmoidGradients,
            'TanSigmoid': Sigmoid.TanSigmoidGradients,
            'ReLU': Sigmoid.ReLUGradients,
        }
        self.OutputFunction = {
            'SoftMax': SoftMax.SoftMax,
            'PureLin': PureLin.PureLin,
        }
        self.LossFunction = {
            'CrossEntropy':CrossEntropy.CrossEntropy,
            'SquaredError': SquaredError.SquaredError,
        }
        self.LossAndOutputGradients = {
            'CrossEntropyWithSoftMax': CrossEntropy.CrossEntropyWithSoftMaxGradients
        }
        self.LossGradients = {
            'SquaredError': SquaredError.SquaredErrorGradients,
        }
        self.OutputGradients = {
            'PureLin': PureLin.PureLinGradients
        }


    def FeedForward(self,data):
        layersOutputs=[data]
        data = cuf.IntergrateBiasAndData(data)
        for i in range(0,self.noOfLayers):
            data=self.Activation[self.activationFunctionNames[i]](data, self.weights[i])
            layersOutputs.append(data)
            data = cuf.IntergrateBiasAndData(data)

        return self.OutputFunction[self.outputFunctionName](data, self.weights[self.noOfLayers]), layersOutputs



    def BackProbGradients(self,output, networkOutput, layerOutputs):
        weightGradients=[None]*(self.noOfLayers+1)
        if self.outputFunctionName == "SoftMax" and self.lossFunctionName== "CrossEntropy":
            gradientsWRTActivation=self.LossAndOutputGradients['CrossEntropyWithSoftMax'](networkOutput, output)
            weightGradients[self.noOfLayers]=np.matmul(gradientsWRTActivation,
                                                              np.transpose(
                                                                  cuf.IntergrateBiasAndData(
                                                                      layerOutputs[self.noOfLayers])))
        else:
            gradientsWRTActivation = self.LossGradients[self.lossFunctionName](networkOutput, output)
            gradientsWRTActivation = self.OutputGradients[self.outputFunctionName](networkOutput, gradientsWRTActivation)
            weightGradients[self.noOfLayers] = np.matmul(gradientsWRTActivation,
                                                                np.transpose(
                                                                    cuf.IntergrateBiasAndData(
                                                                        layerOutputs[self.noOfLayers])))
        for i in reversed(range(0,self.noOfLayers)):
            backProbGradient=np.matmul(np.transpose(cuf.DisIntergrateBiasFromWeights(self.weights[i+1])),
                                                                gradientsWRTActivation)
            gradientsWRTActivation=self.ActivationGradients[self.activationFunctionNames[i]](
                                                                layerOutputs[i+1],backProbGradient)
            weightGradients[i]=np.matmul(gradientsWRTActivation,
                                                np.transpose(cuf.IntergrateBiasAndData(layerOutputs[i])))
        return weightGradients


