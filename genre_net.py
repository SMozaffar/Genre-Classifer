import torch
import torch.nn as nn
import torch.nn.functional as F

class Genrefier_CRNN(nn.Module):
    """
    CRNN Model for Music Genre Classification.

    The model combines Convolutional Neural Networks (CNNs) to extract
    features from mel-spectrograms and Recurrent Neural Networks (GRUs)
    to capture temporal patterns.

    The final layer performs multi-label classification using a sigmoid
    activation function, outputting predictions for each genre.

    Attributes:
    ----------
    conv1, conv2, conv3, conv4 : nn.Conv2d
        Convolutional layers for feature extraction.
    bn0, bn1, bn2, bn3, bn4 : nn.BatchNorm2d
        Batch normalization layers for stabilizing the network.
    elu : nn.ELU
        Exponential Linear Unit (ELU) activation function.
    pool1, pool2, pool3, pool4 : nn.MaxPool2d
        MaxPooling layers to downsample feature maps.
    drop1, drop2, drop3, drop4, drop_final : nn.Dropout
        Dropout layers for regularization to prevent overfitting.
    gru1, gru2 : nn.GRU
        Gated Recurrent Units (GRUs) for learning temporal patterns.
    fc : nn.Linear
        Fully connected output layer for multi-label classification.
    """
    def __init__(self):
        super(Genrefier_CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))

        # Batch normalization
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # ELU Activation
        self.elu = nn.ELU()

        # Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.pool4 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        # Dropout layers
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.drop4 = nn.Dropout(0.1)
        self.drop_final = nn.Dropout(0.3)

        # GRU layers
        self.gru1 = nn.GRU(input_size=128, hidden_size=32, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=False)

        # Fully connected output layer
        self.fc = nn.Linear(32, 10)  # 10 genres

    def forward(self, x):
        # Input block
        x = F.pad(x, (0, 37))  # Padding along time axis
        x = self.bn0(x)

        # Conv block 1
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Conv block 2
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Conv block 3
        x = self.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Conv block 4
        x = self.elu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        # Reshape for GRU layers
        x = x.permute(0, 2, 1, 3)  # Permute to (batch, time, freq, channels)
        x = x.view(x.size(0), -1, 128)  # Reshape to (batch, time, features)

        # GRU block 1
        x, _ = self.gru1(x)

        # GRU block 2
        x, _ = self.gru2(x)

        # Final dropout
        x = self.drop_final(x[:, -1, :])  # Take last time step output

        # Output
        #x = torch.sigmoid(self.fc(x))  # Sigmoid or Softmax? for multi-label classification (DO THIS WITH CROSSENTROPYLOSS)
        x = self.fc(x)  # DO THIS FOR BCEWITHLOGITSLOSS
        return x
