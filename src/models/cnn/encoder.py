import torch.nn as nn

class SmallCNN2d(nn.Module):
    def __init__(self, emb_dim, in_features= 119040, in_channels=5):
        """
        Initializes the SmallCNN2d model.

        Args:
            emb_dim (int): The dimension of the embedding layer output.
        """
        super(SmallCNN2d, self).__init__()
        # in_features = 119040
        self.cn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.2),

            nn.Conv2d(64, 128, 3, padding = (1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.25),

            # nn.Conv2d(128, 256, 3, padding = "same"),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(2, stride = 2),
            # nn.BatchNorm2d(256),
            # nn.Dropout(p = 0.4),

            nn.Flatten(),
            nn.Linear(in_features,  emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the SmallCNN2d model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the CNN model.
        """
        output = self.cn(x)
        return output


class SmallCNN1d(nn.Module):
    def __init__(self, emb_dim, in_features= 119040, in_channels=5):
        """
        Initializes the SmallCNN1d model.

        Args:
            emb_dim (int): The dimension of the embedding layer output.
        """
        super(SmallCNN1d, self).__init__()
        self.cn = nn.Sequential(
            nn.Conv2d(in_channels, 64, (1,3), padding = (0, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((1,2), stride = (1,2)),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.2),

            nn.Conv2d(64, 128, (1, 3), padding = (0, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2), stride = (1, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.25),

            nn.Flatten(),
            nn.Linear(in_features,  emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the SmallCNN1d model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the CNN model.
        """
        output = self.cn(x)
        return output
