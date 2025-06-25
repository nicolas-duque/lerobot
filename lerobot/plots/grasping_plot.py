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

def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimensifrom dataclasses import asdict, dataclasson
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action
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
    #joint_angles[0] *= -1
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


    fig, axs = plt.subplots(1, 1, figsize=(10, 15))


    box_pos = np.array([[-0.085,0.13],[-0.0475,0.13],[-0.01,0.13],[0.0275,0.13],[0.065,0.13]])
    box_pos_avg_e8 = np.array([[-0.0893, 0.1249],[-0.0468, 0.1293],[-0.01013, 0.1292],[0.03179, 0.1284],[0.0708, 0.1254]])
    box_pos_avg_e9_3 = np.array([[-0.0786, 0.193, 5.8925],[-0.0009, 0.1898, 4.1929],[0.0807, 0.1957, 1.2011],
                                 [-0.0393, 0.1533, 5.8473],[0.0353, 0.1567, 1.4479],[-0.0746, 0.1211, 4.0604],
                                 [-0.0025, 0.1188, 3.8189],[0.0722, 0.12, 0.7847],[-0.0396, 0.0861, 4.738],
                                 [0.0366, 0.0868, -1.0705],[-0.0696, 0.0526, 4.0597],[-0.0021, 0.053, 1.3559],[0.0694, 0.0554, -0.1613]])
    
    eps = int(dataset.meta.total_episodes/13)
    for i in range(13):
        color=colors[i]
        
        avg_y = float(round(np.mean(obs_grasp[i*eps:(i+1)*eps, 1]),4))
        avg_x = float(round(np.mean(obs_grasp[i*eps:(i+1)*eps, 0]),4))
        avg_yaw = float(round(np.mean(grasp_angle[i*eps:(i+1)*eps]),4))
        print([avg_y, avg_x, avg_yaw])

        #dist_to_avg = round(np.linalg.norm([box_pos_avg_e8[i,0]-avg_y, box_pos_avg_e8[i,1]-avg_x]),3)
        #print("Block " +  str(i) + ": ", str(avg_y) + ", " + str(avg_x))
        axs.scatter(obs_grasp[i*eps:(i+1)*eps, 1], obs_grasp[i*eps:(i+1)*eps, 0],color=color)
        axs.scatter(avg_y, avg_x, color=color, marker='x', s=100)

        bottom_left = (avg_y-0.0125,avg_x-0.0125)
        rec = plt.Rectangle(bottom_left,0.025,0.025, ec=color, fc='none', angle = avg_yaw, rotation_point="center")
        axs.add_patch(rec)

        #axs.scatter(box_pos_avg_e8[i, 0], box_pos_avg_e8[i, 1], color=color, marker='s', s=20000, facecolors='none')
        #axs.text(box_pos_avg_e8[i,0],box_pos_avg_e8[i,1]-0.004,str(dist_to_avg),ha='center')


    axs.set_xlabel("Y Position")
    axs.set_ylabel("X Position")
    axs.set_title("Grasping Position")
    axs.set_aspect('equal')
    #axs.set_ylim(0.04,0.22)
    #axs.set_xlim(-0.1,0.1)

    plt.show()



if __name__ == "__main__":
    main()



