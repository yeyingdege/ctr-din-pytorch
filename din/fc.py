import torch.nn as nn
from .dice import Dice


class FCLayer(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 bias, 
                 batch_norm=False,
                 dropout_rate=0., 
                 activation='relu', 
                 use_sigmoid=False, 
                 dice_dim=2):
        super(FCLayer, self).__init__()

        self.use_sigmoid = use_sigmoid

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_size))
        
        # FC -> activation -> dropout
        layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        if activation.lower() == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation.lower() == 'dice':
            assert dice_dim
            layers.append(Dice(hidden_size, dim=dice_dim))
        elif activation.lower() == 'prelu':
            layers.append(nn.PReLU())
        else: # None
            pass
        layers.append(nn.Dropout(p=dropout_rate))

        self.fc = nn.Sequential(*layers)
        if self.use_sigmoid:
            self.output_layer = nn.Sigmoid()
        
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        pass


    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.use_sigmoid else self.fc(x)
        
