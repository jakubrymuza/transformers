import torch
import numpy as np
import torch.nn as nn

from torchaudio.models import Emformer

class EmformerClassifier(nn.Module):
    def __init__(self, num_classes, input_length, input_dim, num_heads = 5, ffn_dim = 128, num_layers = 4, segment_length = 4, dropout = 0):
        super(EmformerClassifier, self).__init__()
        self.input_length = input_length

        self.encoder = Emformer(
            input_dim = input_dim,
            num_heads = num_heads,
            ffn_dim = ffn_dim,
            num_layers = num_layers,
            segment_length = segment_length,
            dropout = dropout,
        )

        self.fc1 = nn.Linear(input_length * input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, inputs):
        lengths = torch.tensor(np.full(inputs.size()[0], self.input_length)).cuda()

        enc_outputs, _ = self.encoder(inputs, lengths)

        outputs = torch.flatten(enc_outputs, 1)
        outputs = self.fc1(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc2(outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs