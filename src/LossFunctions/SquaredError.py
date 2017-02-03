import numpy as np


def SquaredError(outputActivations,targetOutput):
    #todo : Add a dimension check
    return (1.0/2.0)*np.linalg.norm(outputActivations-targetOutput)
def SquaredErrorGradients(outputActivations,targetOutput):
    #todo : Add a dimension check
    return outputActivations-targetOutput