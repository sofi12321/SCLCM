import torch.nn as nn
import torch.nn.functional as F

class ConvNet1D(nn.Module):
    """
    Convolutional network for embedding retrieving.
    Base model for CLOCS method
    """
    def __init__(self, embedding_dim=64, input_dim=1000):
        """
        Prepare the network
        :param embedding_dim: Length of the embedding
        :param input_dim: Length of the input sequence
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.pool = nn.MaxPool1d(2, 2)

        # Prepare layers of 1D convolutions with 7-len kernel
        self.conv1 = nn.Conv1d(1, 4, 7)
        self.bn1 = nn.BatchNorm1d(4)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(4, 16, 7)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv1d(16, 32, 7)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(p=0.1)

        # Calculate shape of the flattened vector
        flatten_dim = input_dim
        for i in range(3):
            flatten_dim = (flatten_dim - 6) / 2
        flatten_dim = int(flatten_dim) * 32

        self.fc1 = nn.Linear(flatten_dim , embedding_dim)

    def forward(self, input_x):
        batch_size = input_x.shape[0]
        views_num = input_x.shape[1]

        x = input_x.reshape(batch_size*views_num, 1, -1)

        x = self.dropout1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool(F.relu(self.bn3(self.conv3(x)))))
        # flatten along channels, leave batch
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.reshape(batch_size, views_num, -1)
        return x
