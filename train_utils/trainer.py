from __future__ import division
from train_utils.models import PoseNDF
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import torch.nn as nn
from dataset.pose_dataset import PoseDataSet
from train_utils.average_meter import AverageMeter
import shutil

class PoseNDF_trainer:

    def __init__(self, config):

        self.device = config['train']['device']
        self.enc_name = 'Raw'
        self.batch_size = config['train']['batch_size']

        self.learning_rate = config['train']['optimizer_param']
        self.model = PoseNDF(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        ##create smpl layer
        self.init_net(config)
        self.ep = 0
        self.min_loss = float('inf')
        if config['train']['continue_train']:
            self.ep = self.load_checkpoint()

        train_dataset = PoseDataSet(data_dir=config['data']['data_dir'],
                                    zero_distance_pose_percentage=config['data']['zero_distance_pose_percentage'],
                                    noise_sigmas=config['data']['noise_sigmas'])
        val_dataset = PoseDataSet(data_dir=config['data']['data_dir'],
                                    zero_distance_pose_percentage=config['data']['zero_distance_pose_percentage'],
                                    noise_sigmas=config['data']['noise_sigmas'])
        self.train_dataset = self.get_loader(train_dataset, num_workers=config['train']['num_worker'])
        self.val_dataset = self.get_loader(val_dataset, num_workers=config['train']['num_worker'])

    def init_net(self, config):
        self.iter_nums = 0

        # create exp name based on experiment params
        self.exp_name = config['experiment']['exp_name']
        loss_type = config['train']['loss_type']

        self.exp_name = f"{self.exp_name}_{config['model']['CanSDF']['act']}_{loss_type}_{config['train']['optimizer_param']}"
        self.exp_path = f"{config['experiment']['root_dir']}/{self.exp_name}/"
        self.checkpoint_path = f"{self.exp_path}checkpoints/"
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary')
        self.loss_func = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
        self.eikonal_weight = float(config['train']['eikonal'])
        self.distance_weight = float(config['train']['dist'])

    def train_model(self, ep=None):

        self.model.train()
        epoch_loss = AverageMeter()
        loss = 0.0
        for i, (poses, labels) in enumerate(self.train_dataset):
            self.optimizer.zero_grad()
            poses.requires_grad = True
            dist_pred = self.model(poses)
            grads_wrt_input, = torch.autograd.grad(dist_pred, poses, grad_outputs=dist_pred.data.new(dist_pred.shape).fill_(1),
                                                    create_graph=True)
            dist_gt = labels.to(device=self.device)
            loss = self.loss_func(dist_pred[:, 0], dist_gt) * self.distance_weight
            loss += ((torch.norm(grads_wrt_input.view(self.batch_size, -1), dim=1) - 1) ** 2).mean() * self.eikonal_weight
            
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss, self.batch_size)
            self.iter_nums += 1

        self.writer.add_scalar("train/loss_dist", loss.item(), self.iter_nums)
        self.writer.add_scalar("train/epoch", epoch_loss.avg, ep)
        return loss.item(), epoch_loss.avg

    def inference(self, epoch):
        self.model.eval()
        sum_val_loss = 0
        val_data_loader = self.val_dataset
        out_path = os.path.join(self.exp_path, f'latest_{epoch}')
        os.makedirs(out_path, exist_ok=True)
        for (poses, labels) in val_data_loader:
            dist_pred = self.model(poses)
            dist_gt = labels.to(device=self.device)
            loss = self.loss_func(dist_pred[:, 0], dist_gt)
            sum_val_loss += loss.item()
        val_loss = sum_val_loss / len(val_data_loader)
        self.writer.add_scalar("validation_test/epoch", val_loss, epoch)

        return val_loss

    def validate(self, epoch):
        self.model.eval()
        sum_val_loss = 0
        val_data_loader = self.val_dataset
        for (poses, dist)  in val_data_loader:
            dist_pred = self.model(poses)
            dist_gt = dist.to(device=self.device)
            loss = self.loss_func(dist_pred[:, 0], dist_gt)
            sum_val_loss += loss.item()
        val_loss = sum_val_loss / len(val_data_loader)
        self.writer.add_scalar("validation_vert/epoch", val_loss, epoch)

        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.save_checkpoint(epoch)
        print(f'validation vertices loss at {epoch}....{val_loss:08f}')
        return val_loss

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,
                       _use_new_zipfile_serialization=False)
        else:
            shutil.copyfile(path, path.replace('best', 'previous'))
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,
                       _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        return epoch

    def get_loader(self, dataset, num_workers, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True)
