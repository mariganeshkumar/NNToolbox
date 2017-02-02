from Activations import Sigmoid

class Network:
    def __init__(self, layers,activationFunction,outputFunction,lossFunction):
        self.layers = layers
        self.activationFunction = activationFunction
        self.outputFunction = outputFunction
        self.lossFunction=lossFunction

    ActivationWithBias = {
      'Sigmoid.LogSigmoidWithBias': Sigmoid.LogSigmoidWithBias,
      'Sigmoid.TanSigmoidWithBias': Sigmoid.TanSigmoidWithBias,
    }
    ActivationGradientsWithBias = {
      'Sigmoid.LogSigmoidWithBias': Sigmoid.LogSigmoidGradientsWithBias,
      'Sigmoid.TanSigmoidWithBias': Sigmoid.TanSigmoidGradientsWithBias,
    }
