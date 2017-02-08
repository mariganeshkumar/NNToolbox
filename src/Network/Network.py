import numpy as np

from src.Activations import Sigmoid
from src.LossFunctions import CrossEntropy,SquaredError
from src.OutputFunctions import SoftMax,PureLin
from src.Utility import CommonUtilityFunctions as cuf


class Network:


    def __init__(self, hiddenLayers, activationFunctions, outputFunction, lossFunction, noOfInputs,
                 noOfOutputs,logDir=None):
        self.hiddenLayers = hiddenLayers
        self.noOfLayers = len(hiddenLayers)
        self.activationFunctions = activationFunctions
        self.outputFunction = outputFunction
        self.lossFunction=lossFunction
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
        for i in range(0,self.noOfLayers):
            data=self.Activation[self.activationFunctions[i]](data,self.weights[i])
            layersOutputs.append(data)
            data = cuf.IntergrateBiasAndData(data)

        return  self.OutputFunction[self.outputFunction](data,self.weights[self.noOfLayers]),layersOutputs



    def BackProbGradients(self,output, networkOutput, layerOutputs):
        weightGradients=[None]*(self.noOfLayers+1)
        if self.outputFunction == "SoftMax" and self.lossFunction=="CrossEntropy":
            gradientsWRTActivation=self.LossAndOutputGradients['CrossEntropyWithSoftMax'](networkOutput, output)
            weightGradients[self.noOfLayers]=np.matmul(gradientsWRTActivation,
                                                              np.transpose(
                                                                  cuf.IntergrateBiasAndData(
                                                                      layerOutputs[self.noOfLayers])))
        else:
            gradientsWRTActivation = self.LossGradients[self.lossFunction](networkOutput, output)
            gradientsWRTActivation = self.OutputGradients[self.outputFunction](networkOutput, gradientsWRTActivation)
            weightGradients[self.noOfLayers] = np.matmul(gradientsWRTActivation,
                                                                np.transpose(
                                                                    cuf.IntergrateBiasAndData(
                                                                        layerOutputs[self.noOfLayers])))
        for i in reversed(range(0,self.noOfLayers)):
            backProbGradient=np.matmul(np.transpose(cuf.DisIntergrateBiasFromWeights(self.weights[i+1])),
                                                                gradientsWRTActivation)
            gradientsWRTActivation=self.ActivationGradients[self.activationFunctions[i]](
                                                                layerOutputs[i+1],backProbGradient)
            weightGradients[i]=np.matmul(gradientsWRTActivation,
                                                np.transpose(cuf.IntergrateBiasAndData(layerOutputs[i])))
        return weightGradients



    ####################### Todo : function to be moved out of this class#####################################################

    def WriteLog(self,trainData,trainTragets,step,epoch,lr,valData=None,valTargets=None,testData=None,testTargets=None):
        output,_=self.FeedForward(trainData)
        loss=self.LossFunction[self.lossFunction](output, trainTragets)
        eer = self.accuracy(output,trainTragets)
        filename=self.logDir+'/log_loss_train.txt'
        self.WriteLossLog(epoch,step,loss,lr,filename)
        filename = self.logDir + '/log_err_train.txt'
        self.WriteEERLog(epoch, step, eer, lr, filename)

        if valData!=None:
            output, _ = self.FeedForward(valData)
            loss = self.LossFunction[self.lossFunction](output, valTargets)
            eer = self.accuracy(output, valTargets)
            filename = self.logDir + '/log_loss_valid.txt'
            self.WriteLossLog(epoch, step, loss, lr, filename)
            filename = self.logDir + '/log_err_valid.txt'
            self.WriteEERLog(epoch, step, eer, lr, filename)

        if testData!=None:
            output, _ = self.FeedForward(testData)
            loss = self.LossFunction[self.lossFunction](output, testTargets)
            eer = self.accuracy(output, testTargets)
            filename = self.logDir + '/log_loss_test.txt'
            self.WriteLossLog(epoch, step, loss, lr, filename)
            filename = self.logDir + '/log_err_test.txt'
            self.WriteEERLog(epoch, step, eer, lr, filename)

    def WriteLossLog(self,epoch,step,loss,lr,filename):
        text_file = open(filename, "a+")
        text_file.write("Epoch %s, Step %s, Error: %f, lr: %f \n" % (epoch, step,loss,lr))
        text_file.close()

    def WriteEERLog(self, epoch, step, eer, lr, filename):
        text_file = open(filename, "a+")
        text_file.write("Epoch %s, Step %s, Error: %f, lr: %f \n"  % (epoch, step, eer, lr))
        text_file.close()

    def accuracy(self,predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 0) != np.argmax(labels, 0))
                / predictions.shape[1])


