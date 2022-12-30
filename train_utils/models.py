import torch
import torch.nn as nn
import yaml
from pathlib import Path


class DFNet(nn.Module):

    def __init__(self, config, weight_norm=True):
        super().__init__()
        input_size = config['in_dim']
        hid_layer = config['dims']
        output_size = 1
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            layer = nn.Linear(dims[l], dims[l + 1])

            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            self.layers.append(layer)

    def forward(self, p):

        x = p.reshape(len(p), -1).to(torch.float32)

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


class PoseNDF(nn.Module):

    def __init__(self, config):
        super(PoseNDF, self).__init__()

        self.device = config['train']['device']
        self.dfnet = DFNet(config['model']['CanSDF']).to(self.device)  #TODO: this line is different

    def train(self, mode=True):
        super().train(mode)

    def forward(self, inputs):
        pose = inputs.to(device=self.device)
        rand_pose_in = nn.functional.normalize(pose.to(device=self.device), dim=2)
        dist_pred = self.dfnet(rand_pose_in.reshape(rand_pose_in.shape[0], 84))
        return dist_pred
    
    @classmethod
    def from_checkpoint_dir(cls, checkpoint_path: Path):
        config_path = list(checkpoint_path.glob("*.yaml"))[0]
        with open(str(config_path), 'rb') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        pose_ndf_model = cls(config)
        
        checkpoint_path = checkpoint_path / 'checkpoints' / 'checkpoint_epoch_best.tar'
        checkpoint = torch.load(str(checkpoint_path))
        pose_ndf_model.load_state_dict(checkpoint['model_state_dict'])
        return pose_ndf_model
