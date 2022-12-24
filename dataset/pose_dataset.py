import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_quaternion
from pathlib import Path


class PoseDataSet(Dataset):
    def __init__(self, data_dir: Path, zero_distance_pose_percentage: float = 0.5, noise_sigma: float = 0.01, k_neighbors: int = 5,
                 weighted_sum: bool = False, device='cpu'):
        """Read and load all available pose data from the data dir (AMASS Data set)
           The loaded poses are the 0-set poses and will have a 0 distance.
           
        Args:
            data_dir (Path): _description_
            zero_distance_pose_percentage (float, optional): The percenteage of the data that is the 0-set. Defaults to 1.0.
            noise_sigma (float, optional): When creating a random pose, what is the std of noise to add to an existing pose. 
                The larger the sigma, the further the made up pose can get. Defaults to 0.
            k_neighbors: When creating random poses, for each random pose look for the `k_neighbors` closest poses (in eucledian distance) 
                then for them calculate the mean quaternion distance.
        """
        self.valid_poses = None
        self.device = device
        self.noise_sigma = noise_sigma
        self.pose_weights = torch.tensor([1] * 21) if not weighted_sum else torch.tensor([7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1])
        self.pose_weights = torch.nn.functional.normalize(self.pose_weights.type(torch.float32), dim=0).to(self.device)
        data_files = data_dir.rglob('*.npz')
        for mocap_file in data_files:
            mocap_data = np.load(mocap_file)
            if mocap_data.get('pose_body') is None:
                print(f"File {mocap_file} does not contain `pose_body` data.")
                continue
            pose_data_axis_angel = torch.from_numpy(np.load(mocap_file)['pose_body'].astype(np.float32)).view(-1, 21, 3).to(self.device)
            pose_data_quaternion = axis_angle_to_quaternion(pose_data_axis_angel)
            # TODO: add double cover augmentations
            if self.valid_poses is None:
                self.valid_poses = pose_data_quaternion
            else:
                self.valid_poses = torch.cat((self.valid_poses, pose_data_quaternion), axis=0)
        assert 0 < zero_distance_pose_percentage <= 1, f"`non_zero_pose_percentage` is {zero_distance_pose_percentage} and must be between 0-1" 
        self.length = int(self.valid_poses.shape[0] / zero_distance_pose_percentage)
        
        self.knn = NearestNeighbors(n_neighbors=k_neighbors)
        self.knn.fit(self.valid_poses.cpu().numpy().reshape(self.valid_poses.shape[0], -1))

    def _create_non_zero_pose(self):
        idx = np.random.randint(0, self.valid_poses.shape[0])  # TODO: add seed
        random_pose = self.valid_poses[idx].clone()
        random_pose += torch.normal(0, self.noise_sigma, random_pose.shape)
        random_pose /= np.linalg.norm(random_pose, axis=1, keepdims=True)
        return random_pose
    
    def _calculate_distance_to_zero_set(self, pose_rotations):
        k_nearest_poses_indices = self.knn.kneighbors(pose_rotations.flatten()[None])[1]
        k_nearest_poses = self.valid_poses[k_nearest_poses_indices]
        quaternion_dists = torch.arccos(torch.einsum("bpq,pq->bp", k_nearest_poses, pose_rotations)) ** 2
        quaternion_dists = torch.sqrt(torch.sum(self.pose_weights * quaternion_dists, axis=1))
        return quaternion_dists.mean()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if index < self.valid_poses.shape[0]:
            pose_rotations, distance = self.valid_poses[index], 0
        else:
            pose_rotations = self._create_non_zero_pose()
            distance = self._calculate_distance_to_zero_set(pose_rotations)
        return pose_rotations, distance
        
        
if __name__ == '__main__':
    amass_path = Path("/Users/orlichter/Documents/school/amass/data")
    dataset = PoseDataSet(amass_path)
    dataset[dataset.length-1]
    pass