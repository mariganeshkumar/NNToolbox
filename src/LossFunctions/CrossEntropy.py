import numpy as np


def SquaredError(outputActivations,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivation=#noOfClasses*#noOfExamples
    #todo : Add a dimension check
    return np.mean(np.diag(np.matmul(np.transpose(targetOutput),np.log(outputActivations))))
