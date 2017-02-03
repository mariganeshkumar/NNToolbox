import numpy as np
from src.OutputFunctions import SoftMax

def CrossEntropyWithSoftMaxAndBias(data,weights,biases,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    outputActivations=SoftMax.SoftMaxWithBias(data,weights,biases)
    return -np.mean(np.diag(np.matmul(np.transpose(targetOutput),np.log(outputActivations))))

def CrossEntropyWithSoftMax(data,weights,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    outputActivations=SoftMax.SoftMaxWithBias(data,weights)
    return -np.mean(np.diag(np.matmul(np.transpose(targetOutput),np.log(outputActivations))))

def CrossEntropy(outputActivations,targetOutput):
    #todo : Add a dimension check and set dimension check to false in softmax call and add comment
    return -np.mean(np.diag(np.matmul(np.transpose(targetOutput),np.log(outputActivations))))




def CrossEntropyWithSoftMaxGradients(outputActivations,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    return -(targetOutput-outputActivations)
