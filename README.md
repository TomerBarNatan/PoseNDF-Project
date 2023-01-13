# PoseNDF Implementation

## About
We are implementing [Pose-NDF](https://virtualhumans.mpi-inf.mpg.de/posendf/) paper .The goal of the paper is to create a continuous model for plausible human poses based on neural distance fields. We aim to recreate such a field in order to denoise made up motion capture data into a coherent and plausible movement.

![](https://virtualhumans.mpi-inf.mpg.de/posendf/teaser.png)

## Data Creation 
We use [AMASS](https://amass.is.tue.mpg.de/) dataset. This dataset holds all valid poses (0-distance). A SMPLX pose consists of 21 rotations $\theta_{i\in{0,20}}$. A single pose $\theta=\left(\theta_0, ..., \theta_{20} \right) $ 

In order to create a non-zero poses we do the following:
1. Choose a valid pose at random
2. Add normal distributed noise to it with varying sigmas
3. Project the new pose onto the 4D sphere (as quaternions are on the unit sphere in 4d space)
4. Out of all the valid poses, choose the closest to the random pose in l2 distance.
5. For these 100 calculate actual weighted quaternion distance (not l2).
6. Average the lowest 5 distances we got. Set this as the distance of the random pose to the valid poses.

![](https://user-images.githubusercontent.com/76158808/212336083-8ab281a2-7415-4082-b6de-a243fb50eb72.png)

** We use weighted sum to give more weight to rotations that are closer to the root as these affect their children as well as themselves.

** Notice that SMPLX uses axis angles as rotation representation while we use quaternions. When createing the data we switch to quaternions and when rendering the pose we switch back to axis angle.
## Model Architecture

## Training Process

## Denoising

## Experiments
