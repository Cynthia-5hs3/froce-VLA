import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import GemmaConfig

from force_vla.pi05 import modeling_pi05_base as base
from force_vla.pi05.configuration_pi05_force import ForcePI05Config
from force_vla.pi05.modeling_pi05_force import ForcePI05Model, ForcePI05Policy, load_pi05_backbone
from force_vla.pi05.preparation import ROOT, ForceNormalizer, fit_statistics, split_episodes
from force_vla.pi05.training import TRAINABLE_PREFIXES, configure_trainable, load_delta, save_checkpoint, restore_training_state
from force_vla.pi05.train_pi05_force import load_force_config


def tiny_init(model, config, **kwargs):
    nn.Module.__init__(model)
    model.config = config
    model.gradient_checkpointing_enabled = False
    model.rtc_processor = None
    backbone = base.PaliGemmaWithExpertModel.__new__(base.PaliGemmaWithExpertModel)
    nn.Module.__init__(backbone)
    backbone.freeze_vision_encoder = False
    backbone.train_expert_only = False
    prefix_config = GemmaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=8, num_key_value_heads=1, head_dim=8,
                              vocab_size=128, use_adarms=False)
    expert_config = GemmaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=8, num_key_value_heads=1, head_dim=8,
                              vocab_size=128, use_adarms=True, adarms_cond_dim=64)
    backbone.paligemma = nn.Module()
    backbone.paligemma.config = SimpleNamespace(text_config=prefix_config)
    backbone.paligemma.model = nn.Module()
    backbone.paligemma.model.language_model = base.PiGemmaForCausalLM(prefix_config).model
    backbone.gemma_expert = base.PiGemmaForCausalLM(expert_config)
    backbone.gemma_expert.model.embed_tokens = None
    model.paligemma_with_expert = backbone
    model.action_in_proj = nn.Linear(32, 64)
    model.action_out_proj = nn.Linear(64, 32)
    model.time_mlp_in = nn.Linear(64, 64)
    model.time_mlp_out = nn.Linear(64, 64)


class ForceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.config = ForcePI05Config(device="cpu")
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            self.model = ForcePI05Model(self.config)
        self.tokens = torch.tensor([[1, 2, 3, 0], [3, 2, 1, 4]])
        self.mask = self.tokens != 0
        self.state = torch.randn(2, 10)
        self.history = torch.randn(2, 10, 7)

    @torch.no_grad()
    def test_joint_and_cached_decoder_agree(self):
        self.model.eval()
        noisy = torch.randn(2, 50, 39)
        time = torch.tensor([0.5, 0.8])
        prefix = self.model.embed_prefix([], [], self.tokens, self.mask)
        suffix = self.model.embed_suffix(noisy, time, self.state, self.history)
        pad = torch.cat((prefix[1], suffix[1]), 1)
        blocks = torch.cat((prefix[2], suffix[2]), 1)
        attention = base.make_att_2d_masks(pad, blocks)
        self.assertFalse(attention[:, :4, 4:].any())
        self.assertFalse(attention[:, 4, 5:].any())
        self.assertTrue(attention[:, 5:, 4].all())
        full = self.model._run_expert(*prefix, *suffix)
        expected = torch.cat((self.model.flow_action_out_proj(full), self.model.torque_out_proj(full)), -1)
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, cache = self.model.paligemma_with_expert(
            attention_mask=self.model._prepare_attention_masks_4d(base.make_att_2d_masks(prefix[1], prefix[2])),
            position_ids=prefix[1].cumsum(1) - 1, inputs_embeds=[prefix[0], None], use_cache=True)
        actual = self.model.denoise_step(prefix[1], cache, noisy, time, self.state, self.history)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_policy_backward_and_per_sample_reduction(self):
        config = load_force_config("configs/pi05_force_sft.json")
        config.gradient_checkpointing = False
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            policy = ForcePI05Policy(config)
        configure_trainable(policy, "adapters")
        batch = {"observation.language.tokens": self.tokens, "observation.language.attention_mask": self.mask,
                 "observation.state": self.state, "joint_torque_history": self.history,
                 "future_joint_torque": torch.randn(2, 50, 7), "action": torch.randn(2, 50, 10)}
        with patch.object(policy, "_preprocess_images", return_value=([], [])):
            torch.manual_seed(4)
            loss, _ = policy(batch)
            torch.manual_seed(4)
            samples, _ = policy(batch, reduction="none")
            torch.testing.assert_close(loss, samples.mean())
            loss.backward()
            for prefix in TRAINABLE_PREFIXES:
                gradients = [value.grad for name, value in policy.model.named_parameters() if name.startswith(prefix)]
                self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients), prefix)
            self.assertTrue(all(value.grad is None for value in policy.model.paligemma_with_expert.parameters()))
            output = policy.predict_action_chunk(batch, num_steps=2)
            self.assertEqual(tuple(output.shape), (2, 50, 10))

    def test_checkpoint_seeds_new_action_projections_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = {"model." + name: value.clone().contiguous() for name, value in self.model.state_dict().items()
                      if not name.startswith(TRAINABLE_PREFIXES)}
            source["model.action_in_proj.weight"].fill_(0.25)
            save_file(source, str(Path(directory) / "model.safetensors"))
            load_pi05_backbone(self.model, directory)
            torch.testing.assert_close(self.model.flow_in_proj.weight[:, :32], self.model.action_in_proj.weight)
            source.pop("model.time_mlp_in.weight")
            save_file(source, str(Path(directory) / "model.safetensors"))
            with self.assertRaises(ValueError):
                load_pi05_backbone(self.model, directory)

    def test_split_and_stats_exclude_validation_outlier(self):
        train, validation = split_episodes([1, 1, 2, 3, 3], seed=3)
        self.assertFalse(set(train) & set(validation))
        rows = [{"state": [value] * 10, "future_actions": [value] * 500,
                 "joint_torque_history": [value] * 70, "future_joint_torque": [value] * 350}
                for value in (1., 3., 99999.)]
        statistics = fit_statistics(pa.Table.from_pylist(rows), [0, 1])
        self.assertEqual(statistics["torque"]["offset"], [2.] * 7)
        normalizer = ForceNormalizer(statistics)
        original = torch.randn(2, 50, 10)
        restored = normalizer.transform(normalizer.transform(original, "action"), "action", inverse=True)
        torch.testing.assert_close(original, restored)

    def test_optimizer_scheduler_and_rng_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            model = nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
            model(torch.ones(1, 3)).sum().backward()
            optimizer.step()
            scheduler.step()
            expected = {name: value.detach().clone() for name, value in model.named_parameters()}
            progress = {"step": 1, "epoch": 0, "cursor": 1}
            checkpoint = Path(directory) / "checkpoint"
            save_checkpoint(checkpoint, model, optimizer, scheduler, progress, {"settings": {}})
            random_expected = torch.randn(5)
            for value in model.parameters():
                value.data.zero_()
            optimizer.state.clear()
            restored = restore_training_state(model, optimizer, scheduler, checkpoint)
            self.assertEqual(restored, progress)
            torch.testing.assert_close(torch.randn(5), random_expected)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, expected[name])
                self.assertEqual(optimizer.state[value]["step"].item(), 1)
            self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0009)

    def test_separate_tokens_attention_and_cached_decoder(self):
        self.config.conditioning_layout = "separate"
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            model = ForcePI05Model(self.config).eval()
        noisy = torch.randn(2, 50, 39)
        time = torch.tensor([.5, .8])
        with torch.no_grad():
            prefix = model.embed_prefix([], [], self.tokens, self.mask)
            suffix = model.embed_suffix(noisy, time, self.state, self.history)
            self.assertEqual(suffix[0].shape[1], 52)
            attention = base.make_att_2d_masks(suffix[1], suffix[2])
            self.assertFalse(attention[:, 0, 1:].any())
            self.assertTrue(attention[:, 1, 0].all())
            self.assertFalse(attention[:, 1, 2:].any())
            self.assertTrue(attention[:, 2:, :2].all())
            full = model._run_expert(*prefix, *suffix)
            expected = torch.cat((model.flow_action_out_proj(full), model.torque_out_proj(full)), -1)
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
            _, cache = model.paligemma_with_expert(
                attention_mask=model._prepare_attention_masks_4d(base.make_att_2d_masks(prefix[1], prefix[2])),
                position_ids=prefix[1].cumsum(1) - 1, inputs_embeds=[prefix[0], None], use_cache=True)
            actual = model.denoise_step(prefix[1], cache, noisy, time, self.state, self.history)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_lora_both_transformers_receive_gradients_and_restore(self):
        config = load_force_config("configs/pi05_force_tavla_aligned.json")
        config.gradient_checkpointing = False
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            policy = ForcePI05Policy(config)
        parameters = configure_trainable(policy, "lora")
        self.assertFalse(any("state_torque_fusion" in name for name in parameters))
        self.assertFalse(any("q_proj.weight" in name for name in parameters))
        batch = {"observation.language.tokens": self.tokens, "observation.language.attention_mask": self.mask,
                 "observation.state": self.state, "joint_torque_history": self.history,
                 "future_joint_torque": torch.randn(2, 50, 7), "action": torch.randn(2, 50, 10)}
        optimizer = torch.optim.AdamW(parameters.values(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
        with patch.object(policy, "_preprocess_images", return_value=([], [])):
            loss, _ = policy(batch)
            loss.backward()
        for family in ("paligemma.model.language_model", "gemma_expert.model"):
            gradients = [value.grad for name, value in parameters.items() if family in name and name.endswith("lora_up")]
            self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients), family)
        optimizer.step()
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            checkpoint = Path(directory) / "lora"
            expected = {name: value.detach().clone() for name, value in parameters.items()}
            save_checkpoint(checkpoint, policy, optimizer, scheduler, {"step": 1}, {"settings": {}})
            with torch.no_grad():
                for value in parameters.values():
                    value.zero_()
            load_delta(policy, checkpoint)
            for name, value in parameters.items():
                torch.testing.assert_close(value, expected[name])


if __name__ == "__main__":
    unittest.main()
