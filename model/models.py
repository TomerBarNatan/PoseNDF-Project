import torch.nn as nn


class DFNet(nn.Module):

    def __init__(self, config, weight_norm=True):
        super().__init__()
        input_size = config['in_dim']
        hid_layer = config['dims']
        output_size = 1
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)
        self.activation = nn.ReLU()
        self.layers = []

        for l in range(self.num_layers - 1):
            layer = nn.Linear(dims[l], dims[l + 1])

            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            self.layers.append(layer)

    def forward(self, p):

        x = p.reshape(len(p), -1)

        for l in range(self.num_layers - 1):
            layer = self.layers[l]
            x = layer(x)
            x = self.activation(x)
        return x


class BoneMLP(nn.Module):
    """from LEAP code(CVPR21, Marko et al)"""

    def __init__(self, bone_dim, bone_feature_dim, parent=-1):
        super(BoneMLP, self).__init__()
        if parent == -1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, bone_feature_dim),
            nn.ReLU()
        )

    def forward(self, bone_feat):

        return self.net(bone_feat)
