"""Local PI0.5 implementation with causal joint-torque conditioning.

``modeling_pi05_base.py`` is a copied Evo-RLT/LeRobot PI0.5 implementation.
This module subclasses its core so the original workspace and site-packages
remain untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_pi05_force import ForcePI05Config
from .modeling_pi05_base import (
    PI05Pytorch,
    PI05Policy,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
)
from .torque_adapter import TorqueAdapter


class ForcePI05Model(PI05Pytorch):
    """PI0.5 action expert conditioned on state and ten torque samples."""

    def __init__(self, config: ForcePI05Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        width = self.action_in_proj.out_features
        flow_dim = config.max_action_dim + config.torque_dim
        self.torque_adapter = TorqueAdapter(
            history_steps=config.torque_history_steps,
            torque_dim=config.torque_dim,
            token_dim=width,
        )
        self.state_proj = nn.Linear(config.state_dim, width)
        self.state_torque_fusion = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        ) if config.conditioning_layout == "fused" else None
        self.flow_in_proj = nn.Linear(flow_dim, width)
        self.flow_action_out_proj = nn.Linear(width, config.max_action_dim)
        self.torque_out_proj = nn.Linear(width, config.torque_dim)
        self._initialize_flow_projections()
        self.action_in_proj.requires_grad_(False)
        self.action_out_proj.requires_grad_(False)

    def _initialize_flow_projections(self) -> None:
        """Seed the action portion from PI0.5 and leave torque additions trainable."""
        with torch.no_grad():
            self.flow_in_proj.weight[:, : self.config.max_action_dim].copy_(self.action_in_proj.weight)
            self.flow_in_proj.bias.copy_(self.action_in_proj.bias)
            self.flow_in_proj.weight[:, self.config.max_action_dim :].zero_()
            self.flow_action_out_proj.weight.copy_(self.action_out_proj.weight)
            self.flow_action_out_proj.bias.copy_(self.action_out_proj.bias)

    def _validate_conditioning(self, state: Tensor, torque_history: Tensor) -> None:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must have shape [B, {self.config.state_dim}], got {tuple(state.shape)}"
            )
        expected = (self.config.torque_history_steps, self.config.torque_dim)
        if torque_history.ndim != 3 or tuple(torque_history.shape[1:]) != expected:
            raise ValueError(
                f"torque_history must have shape [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(torque_history.shape)}"
            )
        if state.shape[0] != torque_history.shape[0] or not torch.isfinite(state).all() or not torch.isfinite(torque_history).all():
            raise ValueError("state and torque_history must have matching finite batches")

    def _conditioning_token(self, state: Tensor, torque_history: Tensor) -> Tensor:
        self._validate_conditioning(state, torque_history)
        state_token = self.state_proj(state)
        torque_token = self.torque_adapter(torque_history)
        return self.state_torque_fusion(torch.cat((state_token, torque_token), dim=-1))

    def _conditioning_tokens(self, state: Tensor, torque_history: Tensor) -> Tensor:
        if self.config.conditioning_layout == "fused":
            return self._conditioning_token(state, torque_history)[:, None, :]
        self._validate_conditioning(state, torque_history)
        return torch.stack((self.torque_adapter(torque_history), self.state_proj(state)), dim=1)

    def embed_suffix(
        self,
        noisy_actions_and_torque: Tensor,
        timestep: Tensor,
        state: Tensor,
        torque_history: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepend either the legacy fused token or separate torque/state tokens."""
        expected_flow_dim = self.config.max_action_dim + self.config.torque_dim
        if noisy_actions_and_torque.ndim != 3 or noisy_actions_and_torque.shape[-1] != expected_flow_dim:
            raise ValueError(
                f"noisy_actions_and_torque must end in {expected_flow_dim}, "
                f"got {tuple(noisy_actions_and_torque.shape)}"
            )
        if noisy_actions_and_torque.shape[1] != self.config.chunk_size:
            raise ValueError("flow sequence length must equal config.chunk_size")

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        action_torque_emb = self._apply_checkpoint(self.flow_in_proj, noisy_actions_and_torque)

        def time_mlp(time_input: Tensor) -> Tensor:
            value = self.time_mlp_in(time_input)
            value = F.silu(value)
            value = self.time_mlp_out(value)
            return F.silu(value)

        time_emb = self._apply_checkpoint(time_mlp, time_emb)
        condition_token = self._conditioning_tokens(state, torque_history)
        embeddings = torch.cat((condition_token, action_torque_emb), dim=1)
        batch_size = embeddings.shape[0]
        pad_masks = torch.ones(
            batch_size,
            embeddings.shape[1],
            dtype=torch.bool,
            device=embeddings.device,
        )
        att_masks = torch.tensor(
            [*([1] * (condition_token.shape[1] + 1)), *([0] * (self.config.chunk_size - 1))],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )[None, :].expand(batch_size, -1)
        return embeddings, pad_masks, att_masks, time_emb

    def _prepare_flow_target(self, actions: Tensor, future_torque: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.config.chunk_size:
            raise ValueError("actions must have shape [B, chunk_size, action_dim]")
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError("actions must contain exactly the configured physical dimensions")
        expected_torque = (self.config.chunk_size, self.config.torque_dim)
        if future_torque.ndim != 3 or tuple(future_torque.shape[1:]) != expected_torque:
            raise ValueError(
                f"future_torque must have shape [B, {expected_torque[0]}, {expected_torque[1]}]"
            )
        action_target = pad_vector(actions, self.config.max_action_dim)
        if actions.shape[0] != future_torque.shape[0]:
            raise ValueError("action and torque batch sizes differ")
        if not torch.isfinite(actions).all() or not torch.isfinite(future_torque).all():
            raise ValueError("nonfinite flow targets")
        return torch.cat((action_target, future_torque.to(dtype=action_target.dtype)), dim=-1)

    def _run_expert(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        adarms_cond: Tensor,
    ) -> Tensor:
        if self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        pad_masks = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
        att_masks = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_expert(
            prefix: Tensor,
            suffix: Tensor,
            attention: Tensor,
            positions: Tensor,
            conditioning: Tensor,
        ) -> Tensor:
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, conditioning],
            )
            return suffix_out

        output = self._apply_checkpoint(
            forward_expert,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        return output[:, -self.config.chunk_size :].to(dtype=torch.float32)

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
        future_torque: Tensor,
        state: Tensor,
        torque_history: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        target = self._prepare_flow_target(actions, future_torque)
        if noise is None:
            noise = self.sample_noise(target.shape, target.device)
        if noise.shape != target.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match target {tuple(target.shape)}")
        if time is None:
            time = self.sample_time(target.shape[0], target.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target
        u_t = noise - target
        prefix = self.embed_prefix(images, img_masks, tokens, masks)
        suffix = self.embed_suffix(x_t, time, state, torque_history)
        output = self._run_expert(*prefix, *suffix)
        predicted = torch.cat((self.flow_action_out_proj(output), self.torque_out_proj(output)), dim=-1)
        action_dim = self.config.action_dim
        action_loss = F.mse_loss(predicted[..., :action_dim], u_t[..., :action_dim], reduction="none")
        torque_loss = F.mse_loss(predicted[..., -self.config.torque_dim :], u_t[..., -self.config.torque_dim :], reduction="none")
        return {
            "loss": action_loss.mean() + self.config.future_torque_loss_weight * torque_loss.mean(),
            "action_loss": action_loss.mean(),
            "torque_loss": torque_loss.mean(),
            "loss_per_sample": action_loss.mean(dim=(1, 2))
            + self.config.future_torque_loss_weight * torque_loss.mean(dim=(1, 2)),
            "action_loss_per_dim": action_loss.mean(dim=(0, 1)),
            "predicted_flow": predicted,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        state: Tensor,
        torque_history: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if kwargs:
            raise ValueError(f"Unsupported inference options: {sorted(kwargs)}")
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        batch_size = tokens.shape[0]
        flow_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim + self.config.torque_dim)
        x_t = noise if noise is not None else self.sample_noise(flow_shape, tokens.device)
        if tuple(x_t.shape) != flow_shape:
            raise ValueError("inference noise has the wrong shape")
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att),
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        dt = -1.0 / num_steps
        for step in range(num_steps):
            current_time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=tokens.device
            )
            velocity = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                current_time,
                state,
                torque_history,
            )
            x_t = x_t + dt * velocity
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: Any,
        x_t: Tensor,
        timestep: Tensor,
        state: Tensor,
        torque_history: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, state, torque_history
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat((prefix_pad_2d_masks, suffix_att_2d_masks), dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        expert = self.paligemma_with_expert.gemma_expert.model
        expert.config._attn_implementation = "eager"
        suffix_embs = suffix_embs.to(expert.layers[0].self_attn.q_proj.weight.dtype)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(full_att_2d_masks),
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        output = outputs[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return torch.cat((self.flow_action_out_proj(output), self.torque_out_proj(output)), dim=-1)


class ForcePI05Policy(PI05Policy):
    """LeRobot-compatible policy wrapper for the local force model."""

    config_class = ForcePI05Config
    name = "pi05_force"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, device="cpu", **kwargs):
        if kwargs:
            raise ValueError("Use build_force_policy for base initialization; force checkpoints accept device only")
        from .training import restore_for_inference

        policy, _ = restore_for_inference(pretrained_name_or_path, device=device)
        return policy

    def __init__(self, config: ForcePI05Config, **kwargs: Any) -> None:
        del kwargs
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.rtc_processor = None
        self.model = ForcePI05Model(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if config.device is not None:
            self.model.to(config.device)
        self.reset()

    def _custom_batch_value(self, batch: dict[str, Tensor], *names: str) -> Tensor:
        for name in names:
            if name in batch:
                return batch[name]
        raise KeyError(f"force PI0.5 batch is missing one of {names}")

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none")
        images, image_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = batch[OBS_STATE][..., : self.config.state_dim]
        history = self._custom_batch_value(batch, "joint_torque_history", "observation.joint_torque_history")
        future_torque = self._custom_batch_value(batch, "future_joint_torque", "action.future_joint_torque")
        output = self.model(
            images,
            image_masks,
            tokens,
            masks,
            batch[ACTION],
            future_torque=future_torque,
            state=state,
            torque_history=history,
        )
        loss = output["loss"]
        if reduction == "none":
            per_sample = output["loss_per_sample"]
            return per_sample, {"loss": loss.item(), "action_loss": output["action_loss"].item(), "torque_loss": output["torque_loss"].item()}
        return loss, {
            "loss": loss.item(),
            "action_loss": output["action_loss"].item(),
            "torque_loss": output["torque_loss"].item(),
            "loss_per_dim": output["action_loss_per_dim"].detach().cpu().tolist(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        images, image_masks = self._preprocess_images(batch)
        history = self._custom_batch_value(batch, "joint_torque_history", "observation.joint_torque_history")
        state = batch[OBS_STATE][..., : self.config.state_dim]
        output = self.model.sample_actions(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
            torque_history=history,
            **kwargs,
        )
        return output[..., : self.config.action_dim]


def load_pi05_backbone(model: ForcePI05Model, checkpoint_dir: str | Path) -> dict:
    """Require every backbone tensor; stream weights without silently skipping errors."""
    from safetensors import safe_open

    additions = ("torque_adapter.", "state_proj.", "state_torque_fusion.",
                 "flow_in_proj.", "flow_action_out_proj.", "torque_out_proj.")
    target = model.state_dict()
    required = {name: value for name, value in target.items() if not name.startswith(additions)}
    with safe_open(str(Path(checkpoint_dir) / "model.safetensors"), framework="pt") as source:
        source_keys = set(source.keys())
        mapping = {}
        for name, value in required.items():
            key = "model." + name if "model." + name in source_keys else name
            if key not in source_keys and name.endswith("language_model.embed_tokens.weight"):
                key = "model.paligemma_with_expert.paligemma.lm_head.weight"
            if key not in source_keys or tuple(source.get_slice(key).get_shape()) != tuple(value.shape):
                raise ValueError(f"Checkpoint missing or incompatible backbone tensor: {name}")
            mapping[name] = key
        extra = source_keys - set(mapping.values())
        if extra:
            raise ValueError(f"Unexpected checkpoint tensors: {sorted(extra)[:10]}")
        with torch.no_grad():
            for name, key in mapping.items():
                target[name].copy_(source.get_tensor(key))
    model._initialize_flow_projections()
    return {"loaded_tensors": len(mapping), "new_tensors": [name for name in target if name.startswith(additions)],
            "action_projections_seeded_after_loading": True}


__all__ = ["ForcePI05Model", "ForcePI05Policy", "load_pi05_backbone"]
