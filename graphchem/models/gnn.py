from torch.nn import Module, Linear, Embedding
from torch.nn.functional import relu, dropout
from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool,\
    global_max_pool
from torch import cat


class MessagePassingNet(Module):

    def __init__(self, n_features, n_targets):

        super(MessagePassingNet, self).__init__()

        self.conv1 = SAGEConv(n_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = Linear(256, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, n_targets)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = relu(self.conv1(x.float(), edge_index.long()))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = cat([
            global_max_pool(x, batch),
            global_mean_pool(x, batch)
        ], dim=1)

        x = relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = cat([
            global_max_pool(x, batch),
            global_mean_pool(x, batch)
        ], dim=1)

        x = relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = cat([
            global_max_pool(x, batch),
            global_mean_pool(x, batch)
        ], dim=1)

        x = x1 + x2 + x3

        x = relu(self.lin1(x))
        x = dropout(x, p=0.5, training=self.training)
        x = relu(self.lin2(x))
        return self.lin3(x)
