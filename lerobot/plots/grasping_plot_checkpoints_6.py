import logging
from dataclasses import asdict, dataclass
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
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
    #joint_angles[0] *= -1
    joint_angles_rad = np.radians(joint_angles)
    joint_angles_rad[3] -= np.pi/2
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
    return T[:3, 3]  # Extract position (x, y, z)

def compute_r2(q_d, q_a):
    mean_qd = np.mean(q_d)  # Mean of desired trajectory per joint
    sse = np.sum((q_d - q_a) ** 2)  # Sum of squared errors
    sst = np.sum((q_d - mean_qd) ** 2)  # Total variance
    r2 = 1 - (sse / sst)
    return r2


@parser.wrap()
def main(cfg: ControlPipelineConfig):
    
    dh_params_new = [
        {"theta": 0, "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": 0, "d": 0, "a": 0.108347, "alpha": np.pi},
        {"theta": 0, "d": 0, "a": 0.090467, "alpha": 0},
        {"theta": -np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": 0, "d": 0.05815, "a": 0, "alpha": 0},
    ]

    d = [0.0563, 0.0, 0.0, 0.0, 0.05815]
    a = [0.0, 0.108347, 0.090467, 0.0, 0.0]
    alpha = [np.pi/2, np.pi, 0, -np.pi/2, 0]

    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Load dataset
    #   Load videos
    #   Load observations
    #   Load actions

    repos = ["nduque/eval_robustness_e6_3_100k",
             "nduque/eval_robustness_e6_3_120k",
             "nduque/eval_robustness_e6_3_140k",
             "nduque/eval_robustness_e6_3_160k",
             "nduque/eval_robustness_e6_3_180k",
             "nduque/eval_robustness_e6_3_200k",
             "nduque/eval_robustness_e6_3_220k",
             "nduque/eval_robustness_e6_3_240k",
             "nduque/eval_robustness_e6_3_260k"]
    
    order= [list(range(50)), #100k
            list(range(29)) + [49] +list(range(30,50)), #120k
            list(range(32)) + [38,39,32,33,34,35,36,37,40,41,42,43,44,47,48,49,50,51], #140k
            list(range(50)), #160k
            list(range(50)), #180k
            list(range(50)), #200k
            list(range(29)) + [30,29] + list(range(31,50)), #220k
            list(range(50)), #240k
            list(range(50)), #260k
    ]

    # Loop through datasets
    #   Loop through episodes
    #       get grasp idxs for each checkpoint
    #   subplot of grasping idx for each checkpoint

    titles = ["100k","120k","140k","160k","180k","200k","220k","240k","260k"]
    box_pos_avg_e6 = np.array([[-0.0810, 0.1857],[-0.0810+0.0375, 0.1857],[-0.0063, 0.1915],[-0.0063+0.0375, 0.1915],[0.0738, 0.1917],
                               [-0.0810, 0.1857-0.0325],[-0.0810+0.0375, 0.1857-0.0325],[-0.0063, 0.1915-0.0325],[-0.0063+0.0375, 0.1915-0.0325],[0.0738, 0.1917-0.0325],
                               [-0.0893, 0.1274],[-0.0893+0.0375, 0.1274],[-0.0093, 0.1306],[-0.0093+0.0375, 0.1306],[0.0702, 0.1288],
                               [-0.0893, 0.1274-0.0325],[-0.0893+0.0375, 0.1274-0.0325],[-0.0093, 0.1306-0.0325],[-0.0093+0.0375, 0.1306-0.0325],[0.0702, 0.1288-0.0325],
                               [-0.0843, 0.05475],[-0.0843+0.0375, 0.05475],[-0.00845, 0.0624],[-0.00845+0.0375, 0.0624],[0.0685, 0.0583]])

    c_cfg: EvalControlConfig = cfg.control
    obs_grasp = []
    for i,repo in enumerate(repos):
        fig, axs = plt.subplots(1,1, figsize=(10, 15))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # Generate a colormap with 20 colors
        num_colors = 25
        colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
        plt.rc('axes', prop_cycle=cycler('color', colors))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']        

        dataset = LeRobotDataset(repo, root=c_cfg.root)
        observations = dataset.hf_dataset.select_columns("observation.state")['observation.state']
        episodes = list(range(dataset.meta.total_episodes))
        episode_data_index=get_episode_data_index(dataset.meta.episodes)
        eps_grasp = []
        for j in order[i]:
            # Get the episode
            ep_start = int(episode_data_index["from"][episodes[j]])
            ep_end = int(episode_data_index["to"][episodes[j]])

            obs_set = observations[ep_start:ep_end]
            
            grasp_idx = next((j for j, obs in enumerate(obs_set) if obs[-1] < 30), None)

            if grasp_idx is None:
                grasp_idx = next((j for j, obs in enumerate(obs_set) if obs[-1] < 40), None)

            #eps_grasp.append(obs_set[grasp_idx][:3])
            eps_grasp.append(forward_kinematics(obs_set[grasp_idx][-6:],d,a,alpha))

        cp_grasp = np.array(eps_grasp)
        eps = int(dataset.meta.total_episodes/25)

        for k in range(25):
            color=colors[k]
            
            avg_y = np.mean(cp_grasp[k*eps:(k+1)*eps, 1])
            avg_x = np.mean(cp_grasp[k*eps:(k+1)*eps, 0])
            bottom_left = (box_pos_avg_e6[k,0]-0.0125,box_pos_avg_e6[k,1]-0.0125)
            #dist_to_avg = round(np.linalg.norm([box_pos_avg_e8[i,0]-avg_y, box_pos_avg_e8[i,1]-avg_x]),3)
            dist_to_avg = round(np.linalg.norm([box_pos_avg_e6[k,0]-avg_y, box_pos_avg_e6[k,1]-avg_x]),3)
            print("Block " +  str(k) + ": ", str(avg_y) + ", " + str(avg_x))
            axs.scatter(cp_grasp[k*eps:(k+1)*eps, 1], cp_grasp[k*eps:(k+1)*eps, 0],color=color)
            axs.scatter(avg_y, avg_x, color=color, marker='x', s=100)
            rec = plt.Rectangle(bottom_left,0.025,0.025, ec=color, fc='none')
            axs.add_patch(rec)
            axs.text(box_pos_avg_e6[k,0],box_pos_avg_e6[k,1]-0.004,str(dist_to_avg),ha='center')

        

        axs.set_title(titles[i])
        #axs.set_ylim(0.08,0.17)
        #axs.set_xlim(-0.06,0.03)
        axs.set_aspect('equal')
        #axs[p[i][0],p[i][1]].set_box_aspect(1)
        fig.show()



    plt.show()



if __name__ == "__main__":
    main()



