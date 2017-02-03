import numpy as np

from src.Activations import Sigmoid
from src.OutputFunctions import SoftMax,PureLin
from src.Utility import CommonUtilityFunctions as cuf
from src.LossFunctions import CrossEntropy,SquaredError

class Network:


    def __init__(self, hiddenLayers, activationFunctions, outputFunction, lossFunction, noOfInputs, noOfOutputs):
        self.hiddenLayers = hiddenLayers
        self.activationFunctions = activationFunctions
        self.outputFunction = outputFunction
        self.lossFunction=lossFunction
        self.noOfInputs = noOfInputs
        self.noOfOutputs = noOfOutputs
        np.random.seed(seed=1234)
        self.weights=[]
        self.weights.append(np.random.rand(hiddenLayers[0], noOfInputs + 1))
        for i in range(1, len(hiddenLayers)):
            self.weights.append(np.random.rand(hiddenLayers[i], hiddenLayers[i - 1] + 1))
        self.weights.append(np.random.rand(noOfOutputs, hiddenLayers[-1]+1))

        self.Activation = {
            'LogSigmoid': Sigmoid.LogSigmoid,
            'TanSigmoid': Sigmoid.TanSigmoid,
        }
        self.ActivationGradients = {
            'LogSigmoid': Sigmoid.LogSigmoidGradients,
            'TanSigmoid': Sigmoid.TanSigmoidGradients,
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
        for i in range(0,len(self.hiddenLayers)):
            data=self.Activation[self.activationFunctions[i]](data,self.weights[i])
            layersOutputs.append(data)
            data = cuf.IntergrateBiasAndData(data)

        return  self.OutputFunction[self.outputFunction](data,self.weights[len(self.hiddenLayers)]),layersOutputs



    def BackProbGradients(self,data,output):
        networkOuput,layerOutputs=self.FeedForward(data)
        weightGradients=[None]*(len(self.hiddenLayers)+1)
        if self.outputFunction == "SoftMax" and self.lossFunction=="CrossEntropy":
            gradientsWRTActivation=self.LossAndOutputGradients['CrossEntropyWithSoftMax'](networkOuput,output)
            weightGradients[len(self.hiddenLayers)]=np.matmul(gradientsWRTActivation,
                                                              np.transpose(
                                                                  cuf.IntergrateBiasAndData(
                                                                      layerOutputs[len(self.hiddenLayers)])))
        else:
            gradientsWRTActivation = self.LossGradients[self.lossFunction](networkOuput, output)
            gradientsWRTActivation = self.OutputGradients[self.outputFunction](networkOuput,gradientsWRTActivation)
            weightGradients[len(self.hiddenLayers)] = np.matmul(gradientsWRTActivation,
                                                                np.transpose(
                                                                    cuf.IntergrateBiasAndData(
                                                                        layerOutputs[len(self.hiddenLayers)])))
        for i in reversed(range(0,len(self.hiddenLayers))):
            backProbGradient=np.matmul(np.transpose(cuf.DisIntergrateBiasFromWeights(self.weights[i+1])),
                                                                gradientsWRTActivation)
            gradientsWRTActivation=self.ActivationGradients[self.activationFunctions[i]](
                                                                layerOutputs[i+1],backProbGradient)
            weightGradients[i]=np.matmul(gradientsWRTActivation,
                                                np.transpose(cuf.IntergrateBiasAndData(layerOutputs[i])))
        return weightGradients

    def BatchGradientDecent(self,data,output,eta,itr):
        for i in range(0,itr):
            gradients = self.BackProbGradients(data, output)
            for j in range(0,len(self.hiddenLayers)+1):
                self.weights[j]=self.weights[j]-eta*gradients[j]
            currentnetworkop,_=self.FeedForward(data)
            print('Loss:',self.LossFunction[self.lossFunction](currentnetworkop,output))
