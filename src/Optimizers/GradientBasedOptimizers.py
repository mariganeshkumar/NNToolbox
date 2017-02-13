import numpy as np
import pickle



def BatchGradientDecent(net, trainData, trainTargets, eta, itr, valData=None, valTargets=None, testData=None, testTargets=None,annel=False):
    for i in range(0,itr):
        networkOutput,layerOutputs = net.FeedForward(trainData)
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, trainTargets))
        gradients = net.BackProbGradients(trainTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers+1):
            net.weights[j]= net.weights[j] - eta * gradients[j]
        if net.logDir!=None and i%100==0 :
            net.WriteLog(trainData, trainTargets, i, i, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecent(net, trainData, trainTargets,  itr, batchSize, eta=0.5,valData=None, valTargets=None,
                            testData=None, testTargets=None,annel=False,regularization=False,lamda=0.1):
    batchStart=0
    step = 0
    epoch = 0
    aneelCount=0
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
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
            step=0
        networkOutput, layerOutputs = net.FeedForward(batchData)
        print('Loss:', net.LossFunction[net.lossFunctionNameName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            net.weights[j] = net.weights[j] - eta / batchSize * gradients[j]
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecentWithMomentum(net, trainData, trainTargets, itr, batchSize, eta=0.5, gamma=0.5, valData=None,
                                        valTargets=None,testData=None, testTargets=None,annel=False,
                                        regularization=False,lamda=0.1):
    deltaWeights=[None]* (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    aneelCount = 0
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
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount >3:
                        return net
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            if deltaWeights[j] == None:
                deltaWeights[j]= eta / batchSize * gradients[j]
            else:
                deltaWeights[j] = eta / batchSize * gradients[j] + gamma *deltaWeights[j]
            net.weights[j] = net.weights[j] - deltaWeights[j]
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def NestrovAccelaratedGradientDecent(net, trainData, trainTargets, itr, batchSize, eta=0.5, gamma=0.5, valData=None,
                                        valTargets=None,testData=None, testTargets=None,regularization=False,lamda=0.1,
                                     annel=False):
    deltaWeights=[None]* (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    previousEpochValLoss = np.inf
    aneelCount = 0
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
                    eta=eta*(3.0/4.0)
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        oldWeights=net.weights
        for j in range(0, net.noOfLayers + 1):
            if deltaWeights[j] != None:
                net.weights[j] = net.weights[j] - gamma * deltaWeights[j]
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            if deltaWeights[j] == None:
                deltaWeights[j]= eta / batchSize * gradients[j]
            else:
                deltaWeights[j] = eta / batchSize * gradients[j] + gamma *deltaWeights[j]
            net.weights[j] = oldWeights[j] - deltaWeights[j]
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def AdamOptimizer(net, trainData, trainTargets, itr, batchSize, eta=0.5,b1 = 0.9,b2 = 0.999, valData=None,
                                        valTargets=None,testData=None, testTargets=None,annel=False,
                                        regularization=False,lamda=0.1):
    mt=[None]* (net.noOfLayers + 1)
    vt = [None] * (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    aneelCount = 0
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
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j] += lamda * net.weights[j]
            if mt[j] == None:
                mt[j]= (1-b1)* gradients[j]
                vt[j]= (1-b2) * np.square(gradients[j])
            else:
                mt[j] = b1*mt[j]+(1 - b1) * gradients[j]
                vt[j] = b2*vt[j]+(1 - b2) * np.square(gradients[j])
            net.weights[j] = net.weights[j] - (eta/batchSize)* np.multiply((1/np.sqrt(vt[j]+1e-8)), gradients[j])
        if net.logDir != None and step%100==0:
            net.WriteLog(trainData, trainTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def HandleAneeling(net,valData,valTargets,previousEpochValLoss):
    valOuput, _ = net.FeedForward(valData)
    presentValLoss = net.LossFunction[net.lossFunctionName](valOuput, valTargets)
    if presentValLoss < previousEpochValLoss:
        with open("/tmp/nnet_temp.pickle", "wb") as output_file:
            pickle.dump(net, output_file)
        return presentValLoss,None
    else:
        with open("/tmp/nnet_temp.pickle", "rb") as input_file:
            net = pickle.load(input_file)
        return previousEpochValLoss,net
