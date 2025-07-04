#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

import numpy as np

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():

        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    ####################################################################
    del dataset.features["observation.images.front"]
    del dataset.features["observation.images.above"]
    print(dataset.hf_dataset[0])
    episodes=[]
    if dataset.episodes is None:
        episodes = range(dataset.num_episodes)
    else:
        episodes = dataset.episodes
    grasp_vals = get_grasping_idxs(dataset.hf_dataset.select_columns(["observation.state","episode_index"]),episodes)
    
    dataset.hf_dataset = dataset.hf_dataset.map( lambda data: add_fk(data, grasp_vals))

    dataset.meta.features["observation.state"]["shape"] = (dataset.meta.features["observation.state"]["shape"][0], ) #+ EE pos + 3 dice xy pos + yaw
    dataset.meta.features["observation.ee_pos"] = {
                "dtype": "float32",
                "shape": (4,),
                "names": ["x", "y", "z", "yaw"],
            }
    dataset.meta.features["observation.d_pos"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x", "y", "yaw"],
            }

    dataset.meta.stats["observation.state"]["mean"]  = np.mean(dataset.hf_dataset["observation.state"], axis=0)
    dataset.meta.stats["observation.state"]["std"]  = np.std(dataset.hf_dataset["observation.state"], axis=0)
    dataset.meta.stats["observation.state"]["min"]  = np.min(dataset.hf_dataset["observation.state"], axis=0)
    dataset.meta.stats["observation.state"]["max"]  = np.max(dataset.hf_dataset["observation.state"], axis=0)

    dataset.meta.stats["observation.ee_pos"] = {
        "mean": np.mean(dataset.hf_dataset["observation.ee_pos"], axis=0),
        "std": np.std(dataset.hf_dataset["observation.ee_pos"], axis=0),
        "min": np.min(dataset.hf_dataset["observation.ee_pos"], axis=0),
        "max": np.max(dataset.hf_dataset["observation.ee_pos"], axis=0),
    }

    dataset.meta.stats["observation.d_pos"] = {
        "mean": np.mean(dataset.hf_dataset["observation.d_pos"], axis=0),
        "std": np.std(dataset.hf_dataset["observation.d_pos"], axis=0),
        "min": np.min(dataset.hf_dataset["observation.d_pos"], axis=0),
        "max": np.max(dataset.hf_dataset["observation.d_pos"], axis=0),
    }


    print(dataset.hf_dataset[0])


    #####################################################################

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset._datasets[0].meta if hasattr(dataset, "_datasets") else dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # Episode-level split 
    # #################################################################################
    num_eps = len(dataset.episode_data_index["from"])
    val_size = 1#int(num_eps * 0.1)
    g = torch.Generator().manual_seed(cfg.seed or 42)
    shuffled = torch.randperm(num_eps, generator=g).tolist()
    split_episodes = {"train": shuffled[:-val_size], "val": shuffled[-val_size:]}

    # Build datasets and dataloaders
    dataloaders = {}
    for split, eps in split_episodes.items():
        indices = list(
            EpisodeAwareSampler(
                dataset.episode_data_index,
                episode_indices_to_use=eps,
                drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                shuffle=True,
            )
        )
        subset = torch.utils.data.Subset(dataset, indices)
        dataloaders[split] = torch.utils.data.DataLoader(
            subset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=(split == "train"),
        )

    dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "val_loss": AverageMeter("val_loss", ":.4f"),
        "val_s": AverageMeter("val_s", ":.4f"),  # val_s metric for amortized cost of validation per step
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for current_step in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step = current_step + 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0


        #########################################################################################
        is_validation_step = (
            val_dataloader is not None and cfg.validation_freq > 0 and step % cfg.validation_freq == 0
        )

        # Integrate validation loop directly here
        if is_validation_step:
            validation_start_time = time.perf_counter()

            policy.eval()
            total_val_loss = 0.0
            fraction = 0.05  # only sample 8% of val dataset. Hardcoded for simplicity.
            num_batches_to_run = int(fraction * len(val_dataloader))
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_dataloader):
                    if batch_idx >= num_batches_to_run:
                        break
                    for key in val_batch:
                        if isinstance(val_batch[key], torch.Tensor):
                            val_batch[key] = val_batch[key].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=cfg.policy.use_amp):
                        loss, _ = policy.forward(val_batch)
                    if torch.isfinite(loss):
                        total_val_loss += loss.item()

            policy.train()
            avg_val_loss = total_val_loss / num_batches_to_run
            train_tracker.metrics["val_loss"].reset()
            train_tracker.metrics["val_loss"].update(avg_val_loss)

            validation_duration = time.perf_counter() - validation_start_time
            amortized_val_time = validation_duration / cfg.validation_freq

            train_tracker.metrics["val_s"].reset()
            train_tracker.metrics["val_s"].update(amortized_val_time)

        ###########################################################################################

        if is_log_step:
            logging.info(train_tracker)
            #log_message = str(train_tracker)  # Get base log string
            #logging.info(log_message)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages(keys=["loss", "grad_norm", "lr", "update_s", "dataloading_s"])

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")


##################################################################################################
# Function to compute transformation matrix using DH parameters
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# Function to compute joint positions based on DH parameters
def compute_fk(state):
    dh_params = [
        {"theta": 0, "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": -0.136, "d": 0, "a": 0.10935, "alpha": np.pi},
        {"theta": +0.162, "d": 0, "a": 0.10051, "alpha": 0},
        {"theta": -np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": 0, "d": 0.05815, "a": 0, "alpha": 0},
    ]
    
    T = np.eye(4)
    state = np.radians(state)
    state[1] -= 0.136
    state[2] += 0.162
    state[3] -= np.pi/2  # Adjust for the wrist roll
    state[4] *= -1  # Adjust for the gripper orientation
        
    for i, params in enumerate(dh_params):
        Ti = dh_transform(state[i], params["d"], params["a"], params["alpha"])
        T = T@Ti # Multiply transformation matrices

    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    return torch.tensor(T[:3, 3]), torch.tensor([yaw])  # Extract position (x, y, z)
    
def add_fk(data, grasp_vals):
    
        # Calculate FK for each joint
        fk,yaw = compute_fk(data["observation.state"])
        # Create new column
        data["observation.ee_pos"] = torch.cat([fk, yaw], dim=0)
        data["observation.d_pos"] = torch.tensor(grasp_vals[int(data["episode_index"])])
        
        return data

def get_grasping_idxs(dataset,episodes):
    grasp_vals = {}
    for i in episodes:
        ep_vals = dataset.filter(lambda x: x["episode_index"] == i)
        grasp_idx = next((j for j, obs in enumerate(ep_vals["observation.state"]) if obs[-1] < 30), None)

        if grasp_idx is None:
            grasp_idx = next((j for j, obs in enumerate(ep_vals["observation.state"]) if obs[-1] < 40), None)
        
        if grasp_idx is None:
            grasp_vals[i] = [torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)]
        else:
            fk, yaw = compute_fk(ep_vals["observation.state"][grasp_idx])
            grasp_vals[i] = [fk[0], fk[1],yaw[0]]

    return grasp_vals
##################################################################################################  
if __name__ == "__main__":
    init_logging()
    train()
