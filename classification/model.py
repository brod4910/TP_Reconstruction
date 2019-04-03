import torch
from utils.utils import create_model

class Model(torch.nn.Module):

    def __init__(self, cfg_file, checkpoint= False):
        super(Model, self).__init__()
        self.checkpoint = checkpoint
        self.feature_layers, self.classifier = create_model(cfg_file)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # self.module_wrapper = ModuleWrapperIgnores2ndArg(self.feature_layers)

    def forward(self, input_var):
        # if self.checkpoint is True:
        #     # x = checkpoint(self.module_wrapper,x,self.dummy_tensor)
        #     input_var = checkpoint(self.module_wrapper, input_var, self.dummy_tensor)
        # else:
        input_var = self.feature_layers(input_var)

        input_var = input_var.view(input_var.size(0), -1)
        input_var = self.classifier(input_var)
        return input_var