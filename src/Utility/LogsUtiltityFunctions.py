
from src.Utility.CommonUtilityFunctions import accuracy


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
        output, _ = net.FeedForward(valData)
        loss = net.LossFunction[net.lossFunctionName](output, valTargets)
        eer = accuracy(output, valTargets)
        filename = net.logDir + '/log_loss_valid.txt'
        WriteLossLog(epoch, step, loss, lr, filename)
        filename = net.logDir + '/log_err_valid.txt'
        WriteEERLog(epoch, step, eer, lr, filename)

    if testData!=None :
        output, _ = net.FeedForward(testData)
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

