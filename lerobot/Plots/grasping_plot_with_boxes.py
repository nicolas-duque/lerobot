import logging
from dataclasses import asdict, dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pformat
from copy import copy
import torch
import cv2
from contextlib import nullcontext
import numpy as np
from itertools import accumulate
#import roboticstoolbox as rtb

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    EvalControlConfig,
)
from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs import parser

def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }

# Forward kinematics function
def forward_kinematics(joint_angles,d,a,alpha):
    T = np.eye(4)
    frame = np.eye(4)
    joint_angles_rad = np.radians(joint_angles)
    joint_angles_rad[1] -= 0.136
    joint_angles_rad[2] += 0.162
    joint_angles_rad[3] -= np.pi/2
    joint_angles_rad[4] *= -1 
    for i in range(5):
        theta = joint_angles_rad[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        Ti = np.array([
            [ct, -st * ca, st * sa, a[i] * ct],
            [st, ct * ca, -ct * sa, a[i] * st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        T = T@Ti # Multiply transformation matrices
        
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    return T[:3, 3],yaw#theta_tilt  # Extract position (x, y, z)

def compute_r2(q_d, q_a):
    mean_qd = np.mean(q_d)  # Mean of desired trajectory per joint
    sse = np.sum((q_d - q_a) ** 2)  # Sum of squared errors
    sst = np.sum((q_d - mean_qd) ** 2)  # Total variance
    r2 = 1 - (sse / sst)
    return r2


@parser.wrap()
def main(cfg: ControlPipelineConfig):
    

    colors = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ff7f00",  # Orange
    "#ffff33",  # Yellow
    "#a65628",  # Brown
    "#f781bf",  # Pink
    "#999999",  # Gray
    "#66c2a5",  # Teal
    "#fc8d62",  # Coral
    "#8da0cb",  # Lavender
    "#e78ac3",  # Pinkish Purple
    "#a6d854",  # Lime Green
    "#ffd92f",  # Bright Yellow
    "#e5c494",  # Light Brown
    "#b3b3b3",  # Light Gray
    "#1b9e77",  # Dark Teal
    "#d95f02",  # Dark Orange
    "#7570b3",  # Indigo
    "#e7298a",  # Magenta
    "#66a61e",  # Olive Green
    "#e6ab02",  # Mustard
    "#a6761d",  # Golden Brown
    "#666666",  # Dark Gray
    ]

    d = [0.0563, 0.0, 0.0, 0.0, 0.05815]
    a = [0.0, 0.108347, 0.090467, 0.0, 0.0]
    alpha = [np.pi/2, np.pi, 0, -np.pi/2, 0]

    d = [0.0563, 0.0, 0.0, 0.0, 0.05815]
    a = [0.0, 0.10935, 0.10051, 0.0, 0.0]
    alpha = [np.pi/2, np.pi, 0, -np.pi/2, 0]

    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Load dataset
    #   Load videos
    #   Load observations
    #   Load actions

    c_cfg: EvalControlConfig = cfg.control
    # Create empty dataset or load existing saved episodes
    if c_cfg.episode is not None:
        dataset = LeRobotDataset(c_cfg.repo_id, root=c_cfg.root, episodes=[c_cfg.episode])
        observations = dataset.hf_dataset.select_columns("observation.state")
        actions = dataset.hf_dataset.select_columns("action")
        joints = dataset.meta.info['features']['observation.state']['names']
    else:
        dataset = LeRobotDataset(c_cfg.repo_id, root=c_cfg.root)
        observations = dataset.hf_dataset.select_columns("observation.state")['observation.state']
        actions = dataset.hf_dataset.select_columns("action")['action']
        joints = dataset.meta.info['features']['observation.state']['names']  
        episodes = list(range(dataset.meta.total_episodes))
        episode_data_index=get_episode_data_index(dataset.meta.episodes)

    # Loop through dataset
    #   Get observation (poses)
    #   Save grasping position
    #   Save observation, real action, predicted action in dictionary
    obs_grasp = []
    grasp_angle = []
    for ep_idx in episodes:
        # Get the episode
        ep_start = int(episode_data_index["from"][ep_idx])
        ep_end = int(episode_data_index["to"][ep_idx])
        episode = dataset.hf_dataset

        obs_set = observations[ep_start+30:ep_end]

        if len(obs_set) == 0:
            continue        

        grasp_idx = next((i for i, obs in enumerate(obs_set) if obs[-1] < 30), None)

        if grasp_idx is None:
            grasp_idx = next((i for i, obs in enumerate(obs_set) if obs[-1] < 40), None)

        fk, yaw = forward_kinematics(obs_set[grasp_idx][-6:],d,a,alpha)
        obs_grasp.append(fk)
        grasp_angle.append(yaw)

    
    obs_grasp = np.array(obs_grasp)
    grasp_angle = np.array(grasp_angle)


    fig, axs = plt.subplots(1, 1, figsize=(10, 15))

    axs.scatter(obs_grasp[:,1], obs_grasp[:, 0],color=colors[1])
    for i in range(dataset.meta.total_episodes-1):
        color=colors[0]
        
        bottom_left = (obs_grasp[i,1]-0.0125,obs_grasp[i,0]-0.0125)
        
        print(grasp_angle[i])
        #axs.scatter(obs_grasp[i,1], obs_grasp[i,0], color=color, marker='x', s=100)

        rec = plt.Rectangle(bottom_left,0.025,0.025, ec=color, fc='none', angle = grasp_angle[i], rotation_point="center")
        axs.add_patch(rec)
        #axs.text(box_pos_avg_e6[i,0],box_pos_avg_e6[i,1]-0.004,str(dist_to_avg),ha='center')

    #axs.hist2d(obs_grasp[:, 1],obs_grasp[:, 0], bins=5)


    axs.set_xlabel("Y Position")
    axs.set_ylabel("X Position")
    axs.set_title("Grasping Position")
    axs.set_aspect('equal')
    #axs.set_ylim(0.10,0.15)
    #axs.set_xlim(-0.105,0.085)

    fig.show()



if __name__ == "__main__":
    main()



