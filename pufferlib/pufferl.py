## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import copy
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch

try:
    from pufferlib import _C
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation"
    )

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter

rich.traceback.install(show_locals=False)

import signal  # Aggressively exit on ctrl+c

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None


# ==================== PBB ADDITIONS START ====================
# Preference Network for PBB
class PreferenceNetwork(nn.Module):
    """Per-step preference logit network g_θ(s,a) for PBB"""

    def __init__(self, obs_space, action_space, hidden_size=128, encoder=None):
        super().__init__()

        # Share encoder with policy if provided
        self.shared_encoder = encoder is not None
        if self.shared_encoder:
            self.encoder = encoder
            input_size = hidden_size
        else:
            input_size = np.prod(obs_space.shape)
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
            input_size = hidden_size

        # Concatenate encoded obs with action
        if hasattr(action_space, "n"):
            # Discrete action space
            action_size = action_space.n
            self.discrete = True
        else:
            # Continuous action space
            action_size = np.prod(action_space.shape)
            self.discrete = False

        self.preference_head = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),  # Output single preference logit
        )

    def forward(self, obs, action):
        if self.shared_encoder:
            # Assume obs is already encoded
            encoded = obs
        else:
            batch_size = obs.shape[0]
            encoded = self.encoder(obs.view(batch_size, -1).float())

        # Convert action to one-hot if discrete
        if self.discrete:
            action_one_hot = torch.zeros(
                action.shape[0],
                self.preference_head[0].in_features - encoded.shape[1],
                device=action.device,
            )
            action_one_hot.scatter_(1, action.long().unsqueeze(1), 1)
            action_input = action_one_hot
        else:
            action_input = action.float()

        combined = torch.cat([encoded, action_input], dim=-1)
        return self.preference_head(combined).squeeze(-1)


# Optional Immediate Term Network
class ImmediateTermNetwork(nn.Module):
    """Optional immediate term δ_φ for capturing local events"""

    def __init__(self, obs_space, action_space, hidden_size=64):
        super().__init__()
        input_size = np.prod(obs_space.shape)

        if hasattr(action_space, "n"):
            action_size = action_space.n
            self.discrete = True
        else:
            action_size = np.prod(action_space.shape)
            self.discrete = False

        self.network = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1).float()

        if self.discrete:
            action_one_hot = torch.zeros(
                batch_size,
                self.network[0].in_features - obs_flat.shape[1],
                device=action.device,
            )
            action_one_hot.scatter_(1, action.long().unsqueeze(1), 1)
            action_input = action_one_hot
        else:
            action_input = action.float()

        combined = torch.cat([obs_flat, action_input], dim=-1)
        return self.network(combined).squeeze(-1)


# ==================== PBB ADDITIONS END ====================


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config["seed"]
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config["batch_size"] == "auto" and config["bptt_horizon"] == "auto":
            raise pufferlib.APIUsageError("Must specify batch_size or bptt_horizon")
        elif config["batch_size"] == "auto":
            config["batch_size"] = total_agents * config["bptt_horizon"]
        elif config["bptt_horizon"] == "auto":
            config["bptt_horizon"] = config["batch_size"] // total_agents

        batch_size = config["batch_size"]
        horizon = config["bptt_horizon"]
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f"Total agents {total_agents} <= segments {segments}"
            )

        device = config["device"]
        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == "cuda" and config["cpu_offload"],
            device="cpu" if config["cpu_offload"] else device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *atn_space.shape,
            device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype],
        )
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # LSTM
        if config["use_rnn"]:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {
                i * n: torch.zeros(n, h, device=device)
                for i in range(total_agents // n)
            }
            self.lstm_c = {
                i * n: torch.zeros(n, h, device=device)
                for i in range(total_agents // n)
            }

        # Minibatching & gradient accumulation
        minibatch_size = config["minibatch_size"]
        max_minibatch_size = config["max_minibatch_size"]
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if (
            minibatch_size > max_minibatch_size
            and minibatch_size % max_minibatch_size != 0
        ):
            raise pufferlib.APIUsageError(
                f"minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly"
            )

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}"
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(
            config["update_epochs"] * batch_size / self.minibatch_size
        )
        self.minibatch_segments = self.minibatch_size // horizon
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}"
            )

        # ==================== PBB INITIALIZATION START ====================
        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy

        # Initialize PBB components
        encoder = policy.encoder if hasattr(policy, "encoder") else None
        self.preference_net = PreferenceNetwork(
            obs_space,
            atn_space,
            hidden_size=config.get("pref_hidden_size", 128),
            encoder=encoder,
        ).to(device)

        # Target network for stable TD targets
        self.preference_target = copy.deepcopy(self.preference_net)
        self.preference_target.requires_grad_(False)

        # Optional immediate term network (disabled by default for Breakout)
        self.use_immediate = config.get("use_immediate_term", True)
        if self.use_immediate:
            self.immediate_net = ImmediateTermNetwork(obs_space, atn_space).to(device)
        else:
            self.immediate_net = None

        # Reference policy for preference advantage baseline (DPPO reference)
        self.reference_policy = copy.deepcopy(policy)
        self.reference_policy.requires_grad_(False)

        # PBB hyperparameters
        self.pbb_gamma = config.get("pbb_gamma", 0.99)  # TD discount
        self.pbb_tau = config.get("pbb_tau", 0.005)  # Target network update rate
        self.lambda_td = config.get("lambda_td", 1.0)
        self.lambda_bt = config.get("lambda_bt", 0.85)
        self.lambda_delta = config.get("lambda_delta", 1.0)
        self.lambda_actor = config.get(
            "lambda_actor", 0.1
        )  # Weight for preference advantage loss
        # ==================== PBB INITIALIZATION END ====================

        if config["compile"]:
            self.policy = torch.compile(policy, mode=config["compile_mode"])
            self.policy.forward_eval = torch.compile(
                policy, mode=config["compile_mode"]
            )
            self.preference_net = torch.compile(
                self.preference_net, mode=config["compile_mode"]
            )  # PBB
            pufferlib.pytorch.sample_logits = torch.compile(
                pufferlib.pytorch.sample_logits, mode=config["compile_mode"]
            )
            self.reference_policy = torch.compile(
                self.reference_policy, mode=config["compile_mode"]
            )

        # Optimizer - PBB: now includes preference networks
        params = list(self.policy.parameters()) + list(self.preference_net.parameters())
        if self.immediate_net is not None:
            params += list(self.immediate_net.parameters())

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        elif config["optimizer"] == "muon":
            from heavyball import ForeachMuon

            warnings.filterwarnings(
                action="ignore", category=UserWarning, module=r"heavyball.*"
            )
            import heavyball.utils

            heavyball.utils.compile_mode = (
                config["compile_mode"] if config["compile"] else None
            )
            optimizer = ForeachMuon(
                params,
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config["total_timesteps"] // config["batch_size"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config["precision"]
        self.amp_context = contextlib.nullcontext()
        if config.get("amp", True) and config["device"] == "cuda":
            self.amp_context = torch.amp.autocast(
                device_type="cuda", dtype=getattr(torch, precision)
            )
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(
                f"Invalid precision: {precision}: use float32 or bfloat16"
            )

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (
            time.time() - self.last_log_time
        )

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t  # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)  # , non_blocking=True)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = state["lstm_h"]
                    self.lstm_c[env_id.start] = state["lstm_c"]

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(
                    self.ep_indices[env_id.start].item(),
                    1 + self.ep_indices[env_id.stop - 1].item(),
                )

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = (
                        self.free_idx
                        + torch.arange(num_full, device=config["device"]).int()
                    )
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(
                        action,
                        self.vecenv.action_space.low,
                        self.vecenv.action_space.high,
                    )

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            self.vecenv.send(action)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(
            self.total_agents, device=device, dtype=torch.int32
        )
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    # ==================== PBB HELPER METHOD ====================
    def compute_preference_advantage(self, obs, action):
        """Compute preference advantage for policy gradient"""
        with torch.no_grad():
            # Get current preference value
            if hasattr(self.policy, "encoder"):
                encoded = self.policy.encoder(obs.view(obs.shape[0], -1).float())
                current_pref = self.preference_net(encoded, action)
            else:
                current_pref = self.preference_net(obs, action)

            # Get baseline from reference policy
            state = dict(action=action, lstm_c=None, lstm_h=None)
            ref_logits, _ = self.reference_policy.forward_eval(obs, state)

            # Sample multiple actions for baseline (discrete: exact expectation, continuous: sample)
            if hasattr(self.vecenv.single_action_space, "n"):
                # Discrete: compute exact expectation
                probs = torch.softmax(ref_logits, dim=-1)
                baseline = 0
                for a in range(self.vecenv.single_action_space.n):
                    a_tensor = torch.full((obs.shape[0],), a, device=obs.device)
                    if hasattr(self.policy, "encoder"):
                        a_pref = self.preference_net(encoded, a_tensor)
                    else:
                        a_pref = self.preference_net(obs, a_tensor)
                    baseline += probs[:, a] * a_pref
            else:
                # Continuous: sample actions
                baseline = 0
                n_samples = 4
                for _ in range(n_samples):
                    sampled_action, _, _ = pufferlib.pytorch.sample_logits(ref_logits)
                    if hasattr(self.policy, "encoder"):
                        sample_pref = self.preference_net(encoded, sampled_action)
                    else:
                        sample_pref = self.preference_net(obs, sampled_action)
                    baseline += sample_pref / n_samples

            advantage = current_pref - baseline
            return advantage

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        config = self.config
        device = config["device"]
        horizon = config["bptt_horizon"]

        # DPPO params
        beta = 0.95  # DPO temperature
        percentile = 0.25  # Defines the percentile of "good" and "bad" segments
        reference_update_freq = 5 # Number of epochs before reference is updated

        # Mask segment rewards after terminals
        done = torch.logical_or(self.truncations, self.terminals)
        mask = (done.cumsum(dim=1) == 0) | done
        self.rewards *= mask.float()

        # Compute segment-local discounts for batch usage as an upper triangular matrix
        discount_matrix = torch.triu(
            config["gamma"]
            ** torch.abs(
                torch.arange(horizon, device=device).unsqueeze(1)
                - torch.arange(horizon, device=device)
            )
        )

        # Compute discounted return for all segments
        returns = torch.bmm(
            discount_matrix.unsqueeze(0).expand(self.segments, -1, -1),
            self.rewards.unsqueeze(2),
        ).squeeze(-1)

        # Denote the segment quality as the average return over valid steps
        segment_quality = returns.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Sort the segments, and then create S+ and S-, the indexes of good and bad
        # segments based on the percentile defined
        idxs = torch.argsort(segment_quality)
        n = int(self.segments * percentile)
        S_p, S_m = idxs[-n:], idxs[:n]

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            pairs = self.minibatch_segments // 2
            mb_idx_p = S_p[torch.randint(n, (pairs,), device=device)]
            mb_idx_m = S_m[torch.randint(n, (pairs,), device=device)]

            profile("train_copy", epoch)
            obs_p, actions_p = self.observations[mb_idx_p], self.actions[mb_idx_p]
            obs_m, actions_m = self.observations[mb_idx_m], self.actions[mb_idx_m]

            mask_p, mask_m = mask[mb_idx_p], mask[mb_idx_m]

            profile("train_forward", epoch)

            # ==================== PBB TRAINING START ====================
            # Reshape for processing
            obs_p_flat = obs_p.reshape(-1, *self.vecenv.single_observation_space.shape)
            obs_m_flat = obs_m.reshape(-1, *self.vecenv.single_observation_space.shape)
            # Reshape actions preserving action dimensions
            if len(self.vecenv.single_action_space.shape) == 0:
                # Discrete actions - flatten completely
                actions_p_flat = actions_p.reshape(-1)
                actions_m_flat = actions_m.reshape(-1)
            else:
                # Continuous or multi-dimensional actions
                actions_p_flat = actions_p.reshape(
                    -1, *self.vecenv.single_action_space.shape
                )
                actions_m_flat = actions_m.reshape(
                    -1, *self.vecenv.single_action_space.shape
                )

            # Compute preference logits for winners and losers
            if hasattr(self.policy, "encoder"):
                encoded_p = self.policy.encoder(
                    obs_p_flat.view(obs_p_flat.shape[0], -1).float()
                )
                encoded_m = self.policy.encoder(
                    obs_m_flat.view(obs_m_flat.shape[0], -1).float()
                )
                g_p = self.preference_net(encoded_p, actions_p_flat)
                g_m = self.preference_net(encoded_m, actions_m_flat)
            else:
                g_p = self.preference_net(obs_p_flat, actions_p_flat)
                g_m = self.preference_net(obs_m_flat, actions_m_flat)

            g_p = g_p.reshape(pairs, horizon)
            g_m = g_m.reshape(pairs, horizon)

            # Compute preference differences Δt = g(s+, a+) - g(s-, a-)
            delta_t = g_p - g_m

            # Get target values for TD (with target network)
            with torch.no_grad():
                # Next timestep values
                obs_p_next = obs_p[:, 1:].reshape(
                    -1, *self.vecenv.single_observation_space.shape
                )
                obs_m_next = obs_m[:, 1:].reshape(
                    -1, *self.vecenv.single_observation_space.shape
                )
                actions_p_next = actions_p[:, 1:].reshape(-1)
                actions_m_next = actions_m[:, 1:].reshape(-1)

                if obs_p_next.shape[0] > 0:  # Check if we have next steps
                    if hasattr(self.policy, "encoder"):
                        encoded_p_next = self.policy.encoder(
                            obs_p_next.view(obs_p_next.shape[0], -1).float()
                        )
                        encoded_m_next = self.policy.encoder(
                            obs_m_next.view(obs_m_next.shape[0], -1).float()
                        )
                        g_p_next = self.preference_target(
                            encoded_p_next, actions_p_next
                        )
                        g_m_next = self.preference_target(
                            encoded_m_next, actions_m_next
                        )
                    else:
                        g_p_next = self.preference_target(obs_p_next, actions_p_next)
                        g_m_next = self.preference_target(obs_m_next, actions_m_next)

                    g_p_next = g_p_next.reshape(pairs, horizon - 1)
                    g_m_next = g_m_next.reshape(pairs, horizon - 1)
                    delta_next = g_p_next - g_m_next

                    # Pad with zeros for terminal state
                    delta_next = torch.cat(
                        [delta_next, torch.zeros(pairs, 1, device=device)], dim=1
                    )
                else:
                    delta_next = torch.zeros_like(delta_t)

            # Compute immediate term if enabled
            if self.use_immediate and self.immediate_net is not None:
                delta_immediate = self.immediate_net(
                    obs_p_flat, actions_p_flat
                ) - self.immediate_net(obs_m_flat, actions_m_flat)
                delta_immediate = delta_immediate.reshape(pairs, horizon)
            else:
                delta_immediate = torch.zeros_like(delta_t)

            # TD loss for preference network
            is_not_terminal = torch.arange(horizon, device=device).unsqueeze(0) < (
                horizon - 1
            )
            td_target = (
                delta_immediate.detach()
                + self.pbb_gamma * delta_next * is_not_terminal.float()
            )
            td_loss = ((delta_t - td_target) ** 2 * mask_p).sum() / mask_p.sum().clamp(
                min=1
            )

            # Bradley-Terry anchor loss (segment level)
            segment_logit = (delta_t * mask_p).sum(dim=1)
            bt_loss = nn.functional.binary_cross_entropy_with_logits(
                segment_logit, torch.ones_like(segment_logit)
            )

            # Immediate term loss (if enabled)
            if self.use_immediate and self.immediate_net is not None:
                immediate_target = (
                    delta_t.detach()
                    - self.pbb_gamma * delta_next * is_not_terminal.float()
                )
                immediate_loss = (
                    (delta_immediate - immediate_target) ** 2 * mask_p
                ).sum() / mask_p.sum().clamp(min=1)
            else:
                immediate_loss = torch.tensor(0.0, device=device)

            # Preference advantage policy gradient
            advantages_p = self.compute_preference_advantage(obs_p_flat, actions_p_flat)
            advantages_m = self.compute_preference_advantage(obs_m_flat, actions_m_flat)
            advantages_p = advantages_p.reshape(pairs, horizon)
            advantages_m = advantages_m.reshape(pairs, horizon)

            # Get policy log probs
            if not config["use_rnn"]:
                obs_p_policy = obs_p.reshape(
                    -1, *self.vecenv.single_observation_space.shape
                )
                obs_m_policy = obs_m.reshape(
                    -1, *self.vecenv.single_observation_space.shape
                )
                # Use flattened actions for non-RNN case
                action_p_policy = actions_p_flat
                action_m_policy = actions_m_flat
            else:
                obs_p_policy = obs_p
                obs_m_policy = obs_m
                # Use original actions for RNN case
                action_p_policy = actions_p
                action_m_policy = actions_m

            state_p = dict(lstm_h=None, lstm_c=None)
            state_m = dict(lstm_h=None, lstm_c=None)

            logits_p, _ = self.policy(obs_p_policy, state_p)
            logits_m, _ = self.policy(obs_m_policy, state_m)

            _, logprob_p, ent_p = pufferlib.pytorch.sample_logits(
                logits_p, action=action_p_policy
            )
            _, logprob_m, ent_m = pufferlib.pytorch.sample_logits(
                logits_m, action=action_m_policy
            )

            # Always reshape if logprobs are flattened
            if logprob_p.dim() == 1:
                logprob_p = logprob_p.reshape(pairs, horizon)
                logprob_m = logprob_m.reshape(pairs, horizon)

            # Policy gradient with preference advantage
            actor_loss = -(
                advantages_p * logprob_p * mask_p
            ).sum() / mask_p.sum().clamp(min=1) + (
                advantages_m * logprob_m * mask_m
            ).sum() / mask_m.sum().clamp(
                min=1
            )

            # Also compute original DPO loss for comparison
            with torch.no_grad():
                ref_logits_p, _ = self.reference_policy(obs_p_policy, state_p)
                ref_logits_m, _ = self.reference_policy(obs_m_policy, state_m)

                _, ref_p, _ = pufferlib.pytorch.sample_logits(
                    ref_logits_p, action=action_p_policy
                )
                _, ref_m, _ = pufferlib.pytorch.sample_logits(
                    ref_logits_m, action=action_m_policy
                )

                # Reshape if flattened
                if ref_p.dim() == 1:
                    ref_p = ref_p.reshape(pairs, horizon)
                    ref_m = ref_m.reshape(pairs, horizon)

            r_p = (logprob_p * mask_p).sum(dim=1) - (ref_p * mask_p).sum(dim=1)
            r_m = (logprob_m * mask_m).sum(dim=1) - (ref_m * mask_m).sum(dim=1)
            dpo_loss = -torch.nn.functional.logsigmoid(beta * (r_p - r_m)).mean()

            # Entropy - handle both flattened and shaped cases
            if ent_p.dim() == 1:
                # Entropy is flattened, so flatten masks to match
                mask_p_flat = mask_p.flatten()
                mask_m_flat = mask_m.flatten()
            else:
                # Entropy has same shape as masks
                mask_p_flat = mask_p
                mask_m_flat = mask_m
            mask_sum = mask_p_flat.sum() + mask_m_flat.sum()
            avg_entropy = (
                (ent_p * mask_p_flat).sum() + (ent_m * mask_m_flat).sum()
            ) / mask_sum.clamp(min=1e-8)

            # Total loss (combine DPO with PBB)
            loss = (
                dpo_loss
                - config["ent_coef"] * avg_entropy
                + self.lambda_td * td_loss
                + self.lambda_bt * bt_loss
                + self.lambda_delta * immediate_loss
                + self.lambda_actor * actor_loss
            )
            # ==================== PBB TRAINING END ====================

            # Logging
            profile("train_misc", epoch)
            with torch.no_grad():
                losses["dpo_loss"] += dpo_loss.item() / self.total_minibatches
                losses["entropy"] += avg_entropy.item() / self.total_minibatches
                losses["good_return"] += (
                    segment_quality[S_p].mean().item() / self.total_minibatches
                )
                losses["bad_return"] += (
                    segment_quality[S_m].mean().item() / self.total_minibatches
                )
                # PBB losses
                losses["td_loss"] += td_loss.item() / self.total_minibatches
                losses["bt_loss"] += bt_loss.item() / self.total_minibatches
                losses["immediate_loss"] += (
                    immediate_loss.item() / self.total_minibatches
                )
                losses["actor_loss"] += actor_loss.item() / self.total_minibatches
                losses["pref_adv_mean"] += (
                    advantages_p.mean().item() / self.total_minibatches
                )

            # Learn on accumulated minibatches
            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config["max_grad_norm"]
                )
                torch.nn.utils.clip_grad_norm_(
                    self.preference_net.parameters(), config["max_grad_norm"]
                )
                torch.nn.utils.clip_grad_norm_(
                    self.immediate_net.parameters(), config["max_grad_norm"]
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # ==================== PBB POST-TRAINING UPDATES ====================
        # Update target network with Polyak averaging
        profile("train_misc", epoch)
        with torch.no_grad():
            for param, target_param in zip(
                self.preference_net.parameters(), self.preference_target.parameters()
            ):
                target_param.data.mul_(1 - self.pbb_tau)
                target_param.data.add_(self.pbb_tau * param.data)
        # ==================== END PBB UPDATES ====================

        # Reprioritize experience
        if config["anneal_lr"]:
            self.scheduler.step()

        if self.epoch % reference_update_freq == 0:
            tau = 0.15
            with torch.no_grad():
                for param, ref_param in zip(
                    self.policy.parameters(), self.reference_policy.parameters()
                ):
                    ref_param.data.mul_(1 - tau)
                    ref_param.data.add_(tau * param.data)

        profile.end()

        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if (
            done_training
            or self.global_step == 0
            or time.time() > self.last_log_time + 0.25
        ):
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config["device"]
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            "SPS": dist_sum(self.sps, device),
            "agent_steps": agent_steps,
            "uptime": time.time() - self.start_time,
            "epoch": int(dist_sum(self.epoch, device)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            # **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            # **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self.logger.log(logs, agent_steps)
                return logs
            else:
                return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(
            self.config["data_dir"], f'{self.config["env"]}_{run_id}.pt'
        )
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        # ==================== PBB CHECKPOINT SAVING ====================
        # Save all model components including PBB networks
        checkpoint = {
            "policy": self.uncompiled_policy.state_dict(),
            "preference_net": (
                self.preference_net.state_dict()
                if hasattr(self, "preference_net")
                else None
            ),
            "preference_target": (
                self.preference_target.state_dict()
                if hasattr(self, "preference_target")
                else None
            ),
            "reference_policy": (
                self.reference_policy.state_dict()
                if hasattr(self, "reference_policy")
                else None
            ),
        }
        if hasattr(self, "immediate_net") and self.immediate_net is not None:
            checkpoint["immediate_net"] = self.immediate_net.state_dict()

        torch.save(checkpoint, model_path)
        # ==================== END PBB CHECKPOINT ====================

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": run_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        return model_path

    def print_dashboard(
        self,
        clear=False,
        idx=[0],
        c1="[cyan]",
        c2="[white]",
        b1="[bright_cyan]",
        b2="[bright_white]",
    ):
        config = self.config
        sps = dist_sum(self.sps, config["device"])
        agent_steps = dist_sum(self.global_step, config["device"])
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        profile = self.profile
        console = Console()
        dashboard = Table(
            box=rich.box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="bright_cyan",
        )
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 + PBB {idx[0]*" "}:blowfish:',  # Added PBB indicator
            f"{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%",
            f"{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%",
            f"{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%",
            f"{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%",
        )
        idx[0] = (idx[0] - 1) % 10

        s = Table(box=None, expand=True)
        remaining = "A hair past a freckle"
        if sps != 0:
            remaining = duration(
                (config["total_timesteps"] - agent_steps) / sps, b2, c2
            )

        s.add_column(f"{c1}Summary", justify="left", vertical="top", width=10)
        s.add_column(f"{c1}Value", justify="right", vertical="top", width=14)
        s.add_row(f"{c2}Env", f'{b2}{config["env"]}')
        s.add_row(f"{c2}Params", abbreviate(self.model_size, b2, c2))
        s.add_row(f"{c2}Steps", abbreviate(agent_steps, b2, c2))
        s.add_row(f"{c2}SPS", abbreviate(sps, b2, c2))
        s.add_row(f"{c2}Epoch", f"{b2}{self.epoch}")
        s.add_row(f"{c2}Uptime", duration(self.uptime, b2, c2))
        s.add_row(f"{c2}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf("Evaluate", b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf("  Env", c2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf("Train", b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf("  Learn", c2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.train_misc, b2, c2))

        l = Table(
            box=None,
            expand=True,
        )
        l.add_column(f"{c1}Losses", justify="left", width=16)
        l.add_column(f"{c1}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try:  # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[0;0H" + capture.get())


def abbreviate(num, b2, c2):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return f"{num/1e3:.1f}K"
    elif num < 1e9:
        return f"{num/1e6:.1f}M"
    elif num < 1e12:
        return f"{num/1e9:.1f}B"
    else:
        return f"{num/1e12:.2f}T"


def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return (
        f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s"
        if h
        else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"
    )


def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"], b2, c2), f"{b2}{percent:2d}{c2}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()


class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        if epoch % self.frequency != 0:
            return

        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]["start"] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile["start"]
        profile["elapsed"] += delta
        profile["delta"] += delta

    def end(self):
        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof["delta"] > 0:
                prof["buffer"] = prof["delta"]
                prof["delta"] = 0


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent() / psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                    time.sleep(self.delay)
                    continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def downsample(arr, m):
    if len(arr) < m:
        return arr

    if m == 0:
        return [arr[-1]]

    orig_arr = arr
    last = arr[-1]
    arr = arr[:-1]
    arr = np.array(arr)
    n = len(arr)
    n = (n // m) * m
    arr = arr[-n:]
    downsampled = arr.reshape(m, -1).mean(axis=1)
    return np.concatenate([downsampled, [last]])


class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100 * time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass


class NeptuneLogger:
    def __init__(self, args, load_id=None, mode="async"):
        import neptune as nept

        neptune_name = args["neptune_name"]
        neptune_project = args["neptune_project"]
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def close(self, model_path):
        self.neptune["model"].track_files(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"


class WandbLogger:
    def __init__(self, args, load_id=None, resume="allow"):
        import wandb

        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.wandb = wandb
        self.run_id = wandb.run.id

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def close(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print("World size", world_size)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}"
        )
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args["neptune"]:
        logger = NeptuneLogger(args)
    elif args["wandb"]:
        logger = WandbLogger(args)

    train_config = dict(**args["train"], env=env_name)
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20 * train_config["total_timesteps"]:
                all_logs.append(logs)

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    i = 0
    stats = {}
    while i < 32 or not stats:
        stats = pufferl.evaluate()
        i += 1

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs


def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args["vec"]["backend"]
    if backend != "PufferEnv":
        backend = "Serial"

    args["vec"] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args["save_frames"]:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == "ansi":
            print("\033[0;0H" + render + "\n")
            time.sleep(1 / args["fps"])
        elif driver.render_mode == "rgb_array":
            pass
            # import cv2
            # render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            # cv2.imshow('frame', render)
            # cv2.waitKey(1)
            # time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args["save_frames"]:
            import imageio

            imageio.mimsave(args["gif_path"], frames, fps=args["fps"], loop=0)
            frames.append("Done")


def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb or neptune")

    method = args["sweep"].pop("method")
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(
            f"Invalid sweep method {method}. See pufferlib.sweep"
        )

    sweep = sweep_cls(args["sweep"])
    points_per_run = args["sweep"]["downsample"]
    target_key = f'environment/{args["sweep"]["metric"]}'
    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep.suggest(args)
        total_timesteps = args["train"]["total_timesteps"]
        all_logs = train(env_name, args=args)
        all_logs = [e for e in all_logs if target_key in e]
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = downsample([log["agent_steps"] for log in all_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            args["train"]["total_timesteps"] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args["train"]["total_timesteps"] = total_timesteps


def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args["train"], env=args["env_name"], tag=args["tag"])
    pufferl = PuffeRL(
        train_config, vecenv, policy, neptune=args["neptune"], wandb=args["wandb"]
    )

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])

    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f"Saved {len(weights)} weights to {path}")


def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args["package"]
    module_name = (
        "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    )
    env_module = importlib.import_module(module_name)
    env_name = args["env_name"]
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args["train"]["env_batch_size"])


def load_env(env_name, args):
    package = args["package"]
    module_name = (
        "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    )
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args["env"], **args["vec"])


def load_policy(args, vecenv, env_name=""):
    package = args["package"]
    module_name = (
        "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    )
    env_module = importlib.import_module(module_name)

    device = args["train"]["device"]
    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(vecenv.driver_env, **args["policy"])

    rnn_name = args["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args["rnn_name"])
        policy = rnn_cls(vecenv.driver_env, policy, **args["rnn"])

    policy = policy.to(device)

    load_id = args["load_id"]
    if load_id is not None:
        if args["neptune"]:
            path = NeptuneLogger(args, load_id, mode="read-only").download()
        elif args["wandb"]:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError("No run id provided for eval")

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args["load_model_path"]
    if load_path == "latest":
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        # ==================== PBB CHECKPOINT LOADING ====================
        checkpoint = torch.load(load_path, map_location=device)

        # Handle both old (state_dict) and new (dict with components) format
        if "policy" in checkpoint:
            # New format with PBB components
            state_dict = checkpoint["policy"]
        else:
            # Old format - just the policy state dict
            state_dict = checkpoint

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        # ==================== END PBB LOADING ====================

    return policy


def load_config(env_name):
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint",
    )
    parser.add_argument(
        "--load-id",
        type=str,
        default=None,
        help="Kickstart/eval from from a finished Wandb/Neptune run",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="auto",
        choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"],
    )
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument(
        "--max-runs", type=int, default=200, help="Max number of sweep runs"
    )
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument(
        "--neptune", action="store_true", help="Use neptune for logging"
    )
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Used by torchrun for DDP"
    )
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    args = parser.parse_known_args()[0]

    # Load defaults and config
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")
    if env_name == "default":
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p["base"]["env_name"].split():
                break
        else:
            raise pufferlib.APIUsageError("No config for env_name {}".format(env_name))

    # Dynamic help menu from config
    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    for section in p.sections():
        for key in p[section]:
            fmt = f"--{key}" if section == "base" else f"--{section}.{key}"
            parser.add_argument(
                fmt.replace("_", "-"),
                default=puffer_type(p[section][key]),
                type=puffer_type,
            )

    parser.add_argument(
        "-h",
        "--help",
        default=argparse.SUPPRESS,
        action="help",
        help="Show this help message and exit",
    )

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args["train"]["use_rnn"] = args["rnn_name"] is not None
    return args


def main():
    err = "Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    main()
