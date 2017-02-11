import sys
sys.path.append('src/')

import numpy as np
import argparse
import pickle

from src.Network import Network
from src.MnistHandler import MnistHandler as mh
from src.Optimizers import GradientBasedOptimizers as gbo
from src.Utility import CommonUtilityFunctions as cuf

parser = argparse.ArgumentParser()
parser.add_argument("--lr",help="initial learning rate for gradient descent based algorithms",type=float)
parser.add_argument("--momentum",help="momentum to be used by momentum based algorithms",type=float)
parser.add_argument("--num_hidden",
                    help="number of hidden layers - this does not include the input layer and the output layer",
                    type=int)
parser.add_argument("--sizes",
                    help="a comma separated list for the size of each hidden layer",
                    type=str)
parser.add_argument("--activation",
                    help="the choice of activation function - valid values are tanh/sigmoid",
                    type=str)
parser.add_argument("--loss",
                    help="possible choices are squared error[sq] or cross entropy loss[ce]",
                    type=str)
parser.add_argument("--opt",
                    help="possible choices are adam, nag, gd and momentum",
                    type=str)
parser.add_argument("--batch_size",
                    help="the batch size to be used",
                    type=int)
parser.add_argument("--anneal",
                    help="if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch",
                    type=bool)
parser.add_argument("--save_dir",
                    help="the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network",
                    type=str)
parser.add_argument("--expt_dir",
                    help="the directory in which the log files will be saved",
                    type=str)
parser.add_argument("--mnist",
                    help="path to the mnist data in pickeled format 2",
                    type=str)


#parser.add_argument("--momentum",help="momentum to be used by momentum based algorithms",type=float)

args=parser.parse_args()

if(args.lr==None):
    print("learning rate not provided, using default value of 0.1")
    args.lr=0.1
if(args.momentum==None):
    print("momentum not provided, using default value of 0.1")
    args.momentum=0.1
if(args.num_hidden==None ):
    print("Please provide number of hidden layers and try again. try using option -h for help")
    exit(0)
if(args.sizes==None ):
    print("Please provide size of each hidden layers and try again. try using option -h for help")
    exit(0)
else :
    args.sizes = [int(i) for i in args.sizes.split(',')]
if(args.activation==None ):
    print("Activations of hidden layers not provided. assuming sigmoid by default")
    args.activation="sigmoid"
if args.activation == "sigmoid":
    args.activation="LogSigmoid"
elif args.activation== "tanh":
    args.activation="TanSigmoid"
else :
    print("Invalid activations of hidden layers provided. assuming sigmoid by default")
    args.activation = "LogSigmoid"
args.activation = [args.activation for i in args.sizes]
if(args.loss==None ):
    print("Loss function not provided. try using option -h for help")
    args.activation="sigmoid"
    exit(0)
if args.loss == "sq":
    args.loss= "SquaredError"
else:
    args.loss= "CrossEntropy"

if args.opt not in ['adam','nag','gd','momentum']:
    print("Invalid optimizer provided. try using option -h for help")
    exit(0)

if(args.batch_size==None ):
    print("batch_Size of hidden layers not provided. assuming 200 by default")
    args.batch_size=200

if(args.anneal==None ):
    print("Assuming anneal to false by default")
    args.anneal=False

if(args.expt_dir==None ):
    print("saving log to /tmp by default")
    args.expt_dir="/tmp"

# if(args.size==None ):
#     print("Please provide size of each hidden layers and try again. try using option -h for help")
#     exit(0)


epochs=3


trainData,valData,testData=mh.readMNISTData(args.mnist)
trainLabels=trainData[1]
trainData=np.transpose(trainData[0])
trainTargets = np.transpose(np.eye(len(np.unique(trainLabels)))[trainLabels])

valLabels=valData[1]
valData=np.transpose(valData[0])
valTargets = np.transpose(np.eye(len(np.unique(valLabels)))[valLabels])

testLabels=testData[1]
testData=np.transpose(testData[0])
testTargets = np.transpose(np.eye(len(np.unique(testLabels)))[testLabels])
net = Network.Network(args.sizes,args.activation,'SoftMax',args.loss,trainData.shape[0],trainTargets.shape[0],args.expt_dir)

if args.opt=="nag":
    net=gbo.NestrovAccelaratedGradientDecent(net,trainData,trainTargets,int(trainTargets.shape[1]/args.batch_size)*epochs,args.batch_size,eta=args.lr,gamma=args.momentum,
                                        valData=valData,valTargets=valTargets,testData=testData,testTargets=testTargets,
                                        annel=args.anneal)
validPrediction,_=net.FeedForward(valData)
validPrediction = np.argmax(validPrediction, 0)
testPrediction,_=net.FeedForward(testData)
testPrediction = np.argmax(testPrediction, 0)
outfile = open(args.expt_dir+'/valid_prediction.txt', "w+")
outfile.write("\n".join(map(str,validPrediction)))
outfile.close()
outfile = open(args.expt_dir+'/test_prediction.txt', "w+")
outfile.write("\n".join(map(str,testPrediction)))
outfile.close

if(args.save_dir==None):
    print("No save dir mentioned. Program completed without saving")
else:
    weights= [None] * (len(args.sizes)+1)
    biases = [None] * (len(args.sizes)+1)
    for i in range(0,len(net.weights)):
        weights[i],biases[i]=cuf.DisIntergrateBiasFromWeights(net.weights[i],biasRequired=True)
    with open(args.save_dir+"/model.pickle", "w") as output_file:
        pickle.dump([weights,biases], output_file)