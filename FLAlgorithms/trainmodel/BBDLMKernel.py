
from FLAlgorithms.trainmodel.layers.misc import ModuleWrapper
from FLAlgorithms.trainmodel.BBBLinear import FFGLinear

class BBDLMKernel(ModuleWrapper):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, **kwargs):
        super(BBDLMKernel, self).__init__(**kwargs)
        self.layer1 = FFGLinear(input_dim, hidden_dim)
        self.layer2 = FFGLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        # x = F.tanh(x)
        x = self.layer2(x)
        return x

    def dist(self):
        return self.layer1.dist() + self.layer2.dist()