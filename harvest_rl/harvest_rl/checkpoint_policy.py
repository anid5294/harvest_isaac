"""Strict, deterministic inference for the shipped rsl_rl MLP actor checkpoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F


LAYERS = (("mlp.0", 72, 512), ("mlp.2", 512, 256),
          ("mlp.4", 256, 128), ("mlp.6", 128, 14))


class CheckpointPolicy:
    """Apply the stored observation normalizer and actor mean network, without sampling."""

    def __init__(self, checkpoint, device, *, normalizer_epsilon=1e-2):
        if normalizer_epsilon <= 0:
            raise ValueError("Normalizer epsilon must be positive")
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("actor_state_dict"), dict):
            raise ValueError("Checkpoint has no actor_state_dict")
        state = payload["actor_state_dict"]
        schema = payload.get("observation_schema")
        geometry_contract = {
            'version': 1, 'name': 'parent_capsule_geometry10_v1',
            'actor_dim': 82, 'critic_dim': 88, 'base_actor_dim': 72,
            'base_critic_dim': 78, 'action_dim': 14, 'geometry_dim': 10,
            'geometry_normalization': 'fixed_physical_scaling_only',
            'normalizer_epsilon': .01,
        }
        if schema is not None and schema != geometry_contract:
            raise ValueError("Unsupported observation_schema")
        self.geometry10 = schema is not None
        if self.geometry10 and normalizer_epsilon != .01:
            raise ValueError("Geometry schema requires frozen epsilon .01")
        self.layers = (("mlp.0", 82 if self.geometry10 else 72, 512),) + LAYERS[1:]
        expected = {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std",
                    "obs_normalizer.count", "distribution.std_param"}
        for name, input_size, output_size in self.layers:
            expected.update((f"{name}.weight", f"{name}.bias"))
        if set(state) != expected:
            raise ValueError(f"Unexpected actor keys: missing={sorted(expected - set(state))}, extra={sorted(set(state) - expected)}")
        for name, input_size, output_size in self.layers:
            if state[f"{name}.weight"].shape != (output_size, input_size):
                raise ValueError(f"Unexpected actor shape for {name}.weight")
            if state[f"{name}.bias"].shape != (output_size,):
                raise ValueError(f"Unexpected actor shape for {name}.bias")
        for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            if state[key].shape not in ((72,), (1, 72)):
                raise ValueError(f"Unexpected normalizer shape for {key}")
        if state["obs_normalizer.count"].numel() != 1 or state["distribution.std_param"].numel() != 14:
            raise ValueError("Unexpected normalizer count or action distribution shape")
        for key, value in state.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite actor tensor: {key}")
        if (state["obs_normalizer._std"] < 0).any():
            raise ValueError("Negative observation standard deviation")
        self.state = state
        self.epsilon = normalizer_epsilon
        self.observation_dim = 82 if self.geometry10 else 72
        self.action_dim = 14

    @torch.inference_mode()
    def __call__(self, observation):
        if isinstance(observation, dict):
            observation = observation["policy"]
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"Expected (batch, {self.observation_dim}) policy observation")
        if not torch.isfinite(observation).all():
            raise ValueError("Non-finite policy observation")
        state = self.state
        x = (observation[:, :72] - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + self.epsilon)
        if self.geometry10:
            x = torch.cat((x, observation[:, 72:]), dim=1)
        for index, (name, _, _) in enumerate(self.layers):
            x = F.linear(x, state[f"{name}.weight"], state[f"{name}.bias"])
            if index < len(self.layers) - 1:
                x = F.elu(x)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite actor output")
        return x
