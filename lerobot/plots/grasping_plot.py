import logging
from dataclasses import asdict, dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    
    box_pos_avg_e9_all = [[-0.0809, 0.195, 8.8293],[-0.0415, 0.1988, 8.2231],[-0.0006, 0.1983, 6.3474],[0.0388, 0.1989, 4.407],[0.0805, 0.2004, 2.3005],
                   [-0.0758, 0.1543, 10.03],[-0.0408, 0.1563, 8.1177],[-0.0031, 0.1565, 9.1966],[0.0378, 0.1605, 2.4887],[0.0759, 0.1588, 3.1575],
                   [-0.075, 0.1187, 6.3325],[-0.0374, 0.12, 6.469],[-0.0031, 0.1212, 5.7372],[0.0338, 0.1231, 6.6873],[0.0757, 0.122, 3.5465],
                   [-0.0726, 0.0826, 5.5792],[-0.0389, 0.0837, 5.1754],[-0.0025, 0.0842, 3.2955],[0.0345, 0.0896, 3.8578],[0.071, 0.0891, 2.3324],
                   [-0.0718, 0.0495, 2.1085],[-0.0393, 0.0523, 5.1431],[-0.0042, 0.0541, 3.0332],[0.036, 0.0541, -1.2648],[0.0695, 0.057, -1.611]]
    
    success = np.array([1,1,1,1,1,1,1, #1
                        1,1,1,1,1,0,0, #2
                        1,1,1,1,1,1,1, #3
                        1,1,0,0,1,1,0, #4
                        1,1,1,1,1,1,1, #5
                        0,0,0,1,0,0,0, #6
                        1,1,1,1,1,1,1, #7
                        1,1,0,0,1,1,0, #8
                        1,1,1,1,1,1,1, #9
                        1,1,1,1,0,1,0, #10
                        1,1,1,1,1,1,1, #11
                        0,1,0,1,0,1,0, #12 
                        1,1,1,1,1,1,1, #13 
                        0,0,0,0,1,0,1, #14
                        1,1,1,1,1,1,1, #15
                        1,1,1,0,1,1,0, #16
                        1,1,1,1,1,1,1, #17
                        1,1,1,1,1,1,1, #18
                        1,1,1,1,1,1,1, #19
                        1,0,0,1,0,1,1, #20
                        1,1,1,1,1,1,1, #21
                        0,0,0,0,0,0,0, #22
                        1,1,1,1,1,1,1, #23
                        0,0,0,0,0,0,0, #24
                        1,1,1,1,1,1,1  #25
                        ]) 
    
    eps = int(dataset.meta.total_episodes/25)
    d_xy = []
    box_pos_avg_e9_all = np.array(box_pos_avg_e9_all)
    for i in range(25):
        color=colors[i]

        box_pos = box_pos_avg_e9_all[i]
        
        avg_y = float(round(np.mean(obs_grasp[i*eps:(i+1)*eps, 1]),4))
        avg_x = float(round(np.mean(obs_grasp[i*eps:(i+1)*eps, 0]),4))
        avg_yaw = float(round(np.mean(grasp_angle[i*eps:(i+1)*eps]),4))
        print([avg_y, avg_x, avg_yaw])
        ep_dx = obs_grasp[i*eps:(i+1)*eps, 0] - box_pos_avg_e9_all[i,1]
        ep_dy = obs_grasp[i*eps:(i+1)*eps, 1] - box_pos_avg_e9_all[i,0]
        d_xy.append([ep_dx,ep_dy])
        #dist_to_avg = round(np.linalg.norm([box_pos_avg_e8[i,0]-avg_y, box_pos_avg_e8[i,1]-avg_x]),3)
        #print("Block " +  str(i) + ": ", str(avg_y) + ", " + str(avg_x))
        for j in range(eps):
            if success[i*eps+j] == 0:
                fc = 'none'
            else:
                fc = color
            axs.scatter(obs_grasp[i*eps+j,1], obs_grasp[i*eps+j, 0],color=color, fc=fc)
        #axs.scatter(obs_grasp[i*eps:(i+1)*eps, 1], obs_grasp[i*eps:(i+1)*eps, 0],color=color, fc=)
        axs.scatter(avg_y, avg_x, color=color, marker='x', s=100)

        bottom_left = (box_pos[0]-0.0125,box_pos[1]-0.0125)
        rec = plt.Rectangle(bottom_left,0.025,0.025, ec=color, fc='none', angle = box_pos[2], rotation_point="center")
        axs.add_patch(rec)

        #axs.text(box_pos_avg_e8[i,0],box_pos_avg_e8[i,1]-0.004,str(dist_to_avg),ha='center')
    d_xy = np.array(d_xy)

    axs.set_xlabel("Y Position")
    axs.set_ylabel("X Position")
    axs.set_title("Grasping Position")
    axs.set_aspect('equal')
    #axs.set_ylim(0.04,0.22)
    #axs.set_xlim(-0.1,0.1)

    fig2, axs2 = plt.subplots(5,5, figsize=(15, 15))
    fig3, axs3 = plt.subplots(5,5, figsize=(15, 15))
    fig4, axs4 = plt.subplots(5,5, figsize=(15, 15))

    # Define bin width
    bin_width = 0.005

    for i in range(25):
        # Compute min and max of data rounded to nearest bin edge
        bins = []  # Initialize bins for x and y errors
        for j in range(2):
        # Compute min and max of data rounded to nearest bin edge
            min_edge = np.floor(d_xy[i,j,:].min() / bin_width) * bin_width
            max_edge = np.ceil(d_xy[i,j,:].max() / bin_width) * bin_width
            # Create symmetric bin edges centered on 0
            left = -max(abs(min_edge), abs(max_edge))
            right = -left
            #bins.append(np.arange(left, right + bin_width, bin_width))
            bins.append(np.arange(-0.03, 0.03+bin_width, bin_width))

        axs2[i//5, i%5].hist(d_xy[i,1,:], bins=bins[1])
        axs2[i//5, i%5].axvline(0, color='red', linestyle='--')
        axs2[i//5, i%5].axvline(0.0125, color='green', linestyle='--')
        axs2[i//5, i%5].axvline(-0.01250, color='green', linestyle='--')
        axs3[i//5, i%5].hist(d_xy[i,0,:], bins=bins[0], orientation='horizontal')
        axs3[i//5, i%5].axhline(0, color='red', linestyle='--')
        axs3[i//5, i%5].axhline(0.0125, color='green', linestyle='--')
        axs3[i//5, i%5].axhline(-0.01250, color='green', linestyle='--')
        h = axs4[i//5, i%5].hist2d(d_xy[i,1,:],d_xy[i,0,:], bins=bins)

    fig4.colorbar(h[3], ax=axs4, orientation='horizontal', fraction=0.02, pad=0.1)
    #axs4.set_aspect('equal')

    fig2.suptitle("Horizontal error histogram - Per position")
    fig3.suptitle("Vertical error histogram - Per position")
    fig4.suptitle("2D error histogram - Per position")


    ##########GENERAL ERROR ANALYSIS##########
    bins = np.arange(-0.04, 0.04+bin_width, bin_width)

    fig5, axs5 = plt.subplots(2,2, figsize=(15, 15))
    axs5[0,0].hist(d_xy[:,1,:].flatten(), bins=bins)
    axs5[0,0].set_title("Horizontal error histogram - all tests")
    axs5[0,0].axvline(0.0125, color='green', linestyle='--')
    axs5[0,0].axvline(-0.0125, color='green', linestyle='--')

    axs5[1,0].hist(d_xy[:,0,:].flatten(), bins=bins, orientation='horizontal')
    axs5[1,0].set_title("Vertical error histogram - all tests")
    axs5[1,0].axhline(0.0125, color='green', linestyle='--')
    axs5[1,0].axhline(-0.0125, color='green', linestyle='--')

    h = axs5[0,1].hist2d(d_xy[:,1,:].flatten(), d_xy[:,0,:].flatten(), bins=20)
    axs5[0,1].set_title("2D error histogram - all tests")
    fig5.colorbar(h[3], ax=axs5[0,1], orientation='horizontal', fraction=0.1, pad=0.1)

    d_xy_reshape = d_xy.transpose(0,2,1).reshape(-1, 2)
    axs5[1,1].scatter(
        d_xy_reshape[success == 0, 1],
        d_xy_reshape[success == 0, 0],
        s=15, color='red', label='Failed'
    )

    axs5[1,1].scatter(
        d_xy_reshape[success == 1, 1],
        d_xy_reshape[success == 1, 0],
        s=15, color='blue', label='Success'
    )

    axs5[1,1].legend(loc='upper right')
    axs5[1,1].set_title("2D error scatter - all tests")
    
    ############UNKNOWN ERROR ANALYSIS############
    # Define bin width
    bins = np.arange(-0.04, 0.04+bin_width, bin_width)

    fig6, axs6 = plt.subplots(2,2, figsize=(15, 15))

    d_xy_unknown = d_xy[1::2,:,:]  # Select every second episode (unknown)
    axs6[0,0].hist(d_xy_unknown[:,1,:].flatten(), bins=bins)
    axs6[0,0].set_title("Horizontal error histogram - unknown tests")
    axs6[0,0].axvline(0.0125, color='green', linestyle='--')
    axs6[0,0].axvline(-0.0125, color='green', linestyle='--')

    axs6[1,0].hist(d_xy_unknown[:,0,:].flatten(), bins=bins, orientation='horizontal')
    axs6[1,0].set_title("Vertical error histogram - unknown tests")
    axs6[1,0].axhline(0.0125, color='green', linestyle='--')
    axs6[1,0].axhline(-0.0125, color='green', linestyle='--')

    h = axs6[0,1].hist2d(d_xy_unknown[:,1,:].flatten(), d_xy_unknown[:,0,:].flatten(), bins=20)
    axs6[0,1].set_title("2D error histogram - unknown tests")
    fig6.colorbar(h[3], ax=axs6[0,1], orientation='horizontal', fraction=0.1, pad=0.1)

    d_xy_reshape = d_xy_unknown.transpose(0,2,1).reshape(-1, 2)
    succ_reshaped = success.reshape(-1, 7)
    selected_groups = succ_reshaped[1::2]
    succ_unknown = selected_groups.reshape(-1)
    axs6[1,1].scatter(
        d_xy_reshape[succ_unknown == 0, 1],
        d_xy_reshape[succ_unknown == 0, 0],
        s=15, color='red', label='Failed'
    )

    axs6[1,1].scatter(
        d_xy_reshape[succ_unknown == 1, 1],
        d_xy_reshape[succ_unknown == 1, 0],
        s=15, color='blue', label='Success'
    )

    axs6[1,1].legend(loc='upper right')
    axs6[1,1].set_title("2D error scatter - unknown tests")
    
    
    plt.show()



if __name__ == "__main__":
    main()



