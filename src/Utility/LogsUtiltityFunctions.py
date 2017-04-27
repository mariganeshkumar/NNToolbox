
from src.Utility.CommonUtilityFunctions import accuracy
import numpy as np


def WriteLog(net, trainData, trainTragets, step, epoch, lr, valData=None, valTargets=None, testData=None
             , testTargets=None):
    output , _ =net.FeedForward(trainData)
    loss=net.LossFunction[net.lossFunctionName](output, trainTragets)
    eer = accuracy(output, trainTragets)
    filename= net .logDir + '/log_loss_train.txt'
    WriteLossLog(epoch, step, loss, lr, filename)
    filename = net.logDir + '/log_err_train.txt'
    WriteEERLog(epoch, step, eer, lr, filename)

    if valData!=None :
        output=FeedForwadData(valData,net)
        loss = net.LossFunction[net.lossFunctionName](output, valTargets)
        eer = accuracy(output, valTargets)
        filename = net.logDir + '/log_loss_valid.txt'
        WriteLossLog(epoch, step, loss, lr, filename)
        filename = net.logDir + '/log_err_valid.txt'
        WriteEERLog(epoch, step, eer, lr, filename)

    if testData!=None :
        output = FeedForwadData(testData, net)
        loss = net.LossFunction[net.lossFunctionName](output, testTargets)
        eer = accuracy(output, testTargets)
        filename = net.logDir + '/log_loss_test.txt'
        WriteLossLog(epoch, step, loss, lr, filename)
        filename = net.logDir + '/log_err_test.txt'
        WriteEERLog(epoch, step, eer, lr, filename)

def WriteLossLog(epoch ,step, loss, lr, filename):
    text_file = open(filename, "a+")
    text_file.write("Epoch %s, Step %s, Loss: %f, lr: %f \n" % (epoch, step,loss, lr))
    text_file.close()

def WriteEERLog(epoch, step, eer, lr, filename):
    text_file = open(filename, "a+")
    text_file.write("Epoch %s, Step %s, Error: %f, lr: %f \n"  % ( epoch, step, eer, lr))
    text_file.close()

def FeedForwadData(data,net):
    batchSize=5000
    if data.shape[1] > 5000:
        i = 0
        posteriors = []
        while i + 5000 <= data.shape[1]:
            output, _ = net.FeedForward(data[:, i:i + 5000])
            posteriors.append(output)
            i = i + 5000
        output, _ = net.FeedForward(data[:, i:])
        posteriors.append(output)
        output = np.concatenate(posteriors, axis=1)

    else:
        output, _ = net.FeedForward(data)
    return output

