"""Configuration for the local torque-aware PI0.5 variant."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_force")
@dataclass
class ForcePI05Config(PI05Config):
    """PI0.5 settings plus the force feedback training contract."""

    action_dim: int = 10
    state_dim: int = 10
    torque_dim: int = 7
    torque_history_steps: int = 10
    future_torque_loss_weight: float = 0.1
    use_external_torque: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.action_dim <= self.max_action_dim:
            raise ValueError("invalid action_dim")
        if self.compile_model or self.rtc_config is not None:
            raise ValueError("Force PI05 currently requires compile_model=false and no RTC")
        if self.state_dim < 1 or self.torque_dim < 1 or self.torque_history_steps < 1:
            raise ValueError("state_dim, torque_dim, and torque_history_steps must be positive")
        if self.future_torque_loss_weight < 0:
            raise ValueError("future_torque_loss_weight must be non-negative")
        if self.use_external_torque:
            raise ValueError("external torque is reserved for a separate ablation")
