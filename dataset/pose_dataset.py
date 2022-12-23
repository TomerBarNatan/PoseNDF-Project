import torch
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_quaternion
from pathlib import Path


class PoseDataSet(Dataset):
    def __init__(self, data_dir: Path, zero_distance_pose_percentage: float = 1.0):
        self.valid_poses = None
        data_files = data_dir.rglob('*.npz')
        for mocap_file in data_files:
            mocap_data = np.load(mocap_file)
            if mocap_data.get('pose_body') is None:
                print(f"File {mocap_file} does not contain `pose_body` data.")
                continue
            pose_data_axis_angel = torch.from_numpy(np.load(mocap_file)['pose_body'].astype(np.float32)).view(-1, 21, 3)
            pose_data_quaternion = axis_angle_to_quaternion(pose_data_axis_angel)
            if self.valid_poses is None:
                self.valid_poses = pose_data_quaternion
            else:
                self.valid_poses = torch.cat((self.valid_poses, pose_data_quaternion), axis=0)
        assert 0 < zero_distance_pose_percentage <= 1, f"`non_zero_pose_percentage` is {zero_distance_pose_percentage} and must be between 0-1" 
        self.length = int(self.valid_poses.shape[0] / zero_distance_pose_percentage)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if index < self.valid_poses.shape[0]:
            return (self.valid_poses[index], 0)
        else:
            return self.create_non_zero_pose()
        
        
if __name__ == '__main__':
    amass_path = Path("/Users/orlichter/Documents/school/amass/data")
    dataset = PoseDataSet(amass_path)
    pass