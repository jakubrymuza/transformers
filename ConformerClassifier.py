import torch
import numpy as np
import torch.nn as nn

from torchaudio.models import Conformer

class ConformerClassifier(nn.Module):
    def __init__(self, num_classes, input_length, input_dim, num_heads = 5, ffn_dim = 256, num_layers = 4, dropout = 0, depthwise_conv_kernel_size = 31):
        super(ConformerClassifier, self).__init__()
        self.input_length = input_length

        self.encoder = Conformer(
            input_dim = input_dim,
            num_heads = num_heads,
            ffn_dim = ffn_dim,
            num_layers = num_layers,
            depthwise_conv_kernel_size = depthwise_conv_kernel_size,
            dropout = dropout,
        )
        self.fc = nn.Linear(input_dim, num_classes)


    def forward(self, inputs):
        lengths = torch.tensor(np.full(inputs.size()[0], self.input_length)).cuda()

        enc_outputs, _ = self.encoder(inputs, lengths)
        outputs = self.fc(enc_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs