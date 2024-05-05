import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionClassifierMod(nn.Module):
    def __init__(self, input_shape, nb_classes, nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=10, kernel_size=41, bottleneck_size=32):
        super(InceptionClassifierMod, self).__init__()
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.bottleneck_size = bottleneck_size
        
        self.conv1 = nn.Conv1d(input_shape[0], nb_filters * 4, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='reflect')
        
        inception_modules = [InceptionModule(nb_filters * 4, nb_filters, bottleneck_size, [kernel_size // (2 ** i) for i in range(3)]) 
                             for _ in range(depth)]
        self.inception_modules = nn.Sequential(*inception_modules)
        
        if use_residual:
            self.residual_blocks = nn.ModuleList([ResidualBlock(nb_filters * 4, nb_filters * 4) for _ in range(depth // 3)])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, nb_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.inception_modules(x)
        
        if self.use_residual:
            for i, block in enumerate(self.residual_blocks):
                if (i + 1) % 3 == 0:
                    x = block(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters=32, bottleneck_size=32*4, kernel_sizes=[41]):
        super(InceptionModule, self).__init__()
        self.bottleneck_size = bottleneck_size
        
        if bottleneck_size and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1)
        
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_layers.append(nn.Conv1d(bottleneck_size if bottleneck_size else in_channels,
                                              nb_filters, kernel_size=kernel_size, padding='same'))
        
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv1d(in_channels, nb_filters, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(nb_filters * len(kernel_sizes) + nb_filters)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        input_inception = self.bottleneck(x) if hasattr(self, 'bottleneck') else x
        conv_outputs = [conv_layer(input_inception) for conv_layer in self.conv_layers]
        x = self.max_pool(x)
        x = self.conv_6(x)
        conv_outputs.append(x)
        #[print(x.shape) for x in conv_outputs]
        output = torch.cat(conv_outputs, dim=1)
        output = self.batch_norm(output)
        output = F.relu(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, nb_filters):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(nb_filters)
        
    def forward(self, x):
        shortcut = self.conv(x)
        shortcut = self.batch_norm(shortcut)
        x = x + shortcut
        x = F.relu(x)
        return x


