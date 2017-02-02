import numpy as np


def SquaredError(outputActivations,targetOutput):
    #todo : Add a dimension check
    return np.linalg.norm(outputActivations-targetOutput)
