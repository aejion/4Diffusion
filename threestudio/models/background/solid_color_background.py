from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *


@threestudio.register("solid-color-background")
class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        if not self.training:
            return (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
        )
        return (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
            * self.env_color
        )
