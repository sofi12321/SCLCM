import torch.nn as nn

class Classifier(nn.Module):
    """
    Classification model for embedding models
    """
    def __init__(self,base_model,output_dim,embedding_dim=64):
        """
        Prepare the network
        :param base_model: model returning embedding of shape (batch, channel, embedding_dim)
        :param output_dim: Number of classes
        :param embedding_dim: Length of the embedding
        """
        super().__init__()
        self.base_model = base_model
        self.layer1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.last_layer = nn.Linear(embedding_dim//2, output_dim)

    def forward(self,x):
        # Obtain embedding
        h = self.base_model(x)
        # Classification layers
        h = F.relu(self.layer1(h))
        h = self.last_layer(h)
        return h


def get_sequential_classifier(encoder, num_classes, emb_dim=128):
    return nn.Sequential(
        encoder,
        nn.Linear(emb_dim, 1024),
        nn.LeakyReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p = 0.3),
        nn.Linear(1024, num_classes),
    )
