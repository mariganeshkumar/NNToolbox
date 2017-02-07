import numpy as np
import pickle



def BatchGradientDecent(net, trainData, trainTargets, eta, itr, valData=None, valTargets=None, testData=None, testTargets=None,annel=False):
    for i in range(0,itr):
        networkOutput,layerOutputs = net.FeedForward(trainData)
        print('Loss:', net.LossFunction[net.lossFunction](networkOutput, trainTargets))
        gradients = net.BackProbGradients(trainTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers+1):
            net.weights[j]= net.weights[j] - eta * gradients[j]
        if net.logDir!=None and i%100==0 :
            net.WriteLog(trainData, trainTargets, i, i, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecent(net, trainData, trainTargets,  itr, batchSize, eta=0.5,valData=None, valTargets=None,
                            testData=None, testTargets=None,annel=False):
    batchStart=0
    step = 0
    epoch = 0
    previousEpochValLoss=np.inf
    for i in range(0, itr):
        #batchSelection=np.random.choice(np.arange(trainData.shape[1]), batchSize)
        step=step+1
        batchData=trainData[:,batchStart:batchStart+batchSize]
        batchTargets=trainTargets[:,batchStart:batchStart+batchSize]
        batchStart=batchSize+batchStart
        if(batchStart>=trainData.shape[1]):
            epoch=epoch+1
            batchStart= batchStart-trainData.shape[1]
            if annel and valData !=None:
                previousEpochValLoss,tempNet=HandleAneeling(net,valData,valTargets,previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta/2
            step=0
        networkOutput, layerOutputs = net.FeedForward(batchData)
        print('Loss:', net.LossFunction[net.lossFunction](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            net.weights[j] = net.weights[j] - eta / batchSize * gradients[j]
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecentWithMomentum(net, trainData, trainTargets, itr, batchSize, eta=0.5, gamma=0.5, valData=None,
                                        valTargets=None,testData=None, testTargets=None,annel=False):
    deltaWeights=[None]* (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    previousEpochValLoss = np.inf
    for i in range(0, itr):
        step = step + 1
        batchData = trainData[:, batchStart:batchStart + batchSize]
        batchTargets = trainTargets[:, batchStart:batchStart + batchSize]
        batchStart = batchSize + batchStart
        networkOutput, layerOutputs = net.FeedForward(batchData)
        if (batchStart >= trainData.shape[1]):
            epoch = epoch + 1
            batchStart = batchStart - trainData.shape[1]
            step = 0
            if annel and valData !=None:
                previousEpochValLoss, tempNet = HandleAneeling(net, valData, valTargets, previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta/2
                    gamma=gamma/2
        print('Loss:', net.LossFunction[net.lossFunction](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if deltaWeights[j] == None:
                deltaWeights[j]= eta / batchSize * gradients[j]
            else:
                deltaWeights[j] = eta / batchSize * gradients[j] + gamma *deltaWeights[j]
            net.weights[j] = net.weights[j] - deltaWeights[j]
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def HandleAneeling(net,valData,valTargets,previousEpochValLoss):
    valOuput, _ = net.FeedForward(valData)
    presentValLoss = net.LossFunction[net.lossFunction](valOuput, valTargets)
    if presentValLoss < previousEpochValLoss:
        with open("/tmp/nnet_temp.pickle", "wb") as output_file:
            pickle.dump(net, output_file)
        return presentValLoss,None
    else:
        with open("/tmp/nnet_temp.pickle", "rb") as input_file:
            net = pickle.load(input_file)
        return previousEpochValLoss,net
