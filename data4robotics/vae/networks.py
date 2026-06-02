from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models


def _make_norm(norm_cfg: Dict):
    if norm_cfg["name"] == "batch_norm":
        return nn.BatchNorm2d
    if norm_cfg["name"] == "group_norm":
        num_groups = norm_cfg["num_groups"]
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    if norm_cfg["name"] == "diffusion_policy":

        def _gn_builder(num_channels):
            num_groups = int(num_channels // 16)
            return nn.GroupNorm(num_groups, num_channels)

        return _gn_builder
    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _construct_resnet(size: int, norm, weights: Optional[str] = None):
    if size == 18:
        weight_enum = models.ResNet18_Weights
        backbone = models.resnet18(norm_layer=norm)
    elif size == 34:
        weight_enum = models.ResNet34_Weights
        backbone = models.resnet34(norm_layer=norm)
    elif size == 50:
        weight_enum = models.ResNet50_Weights
        backbone = models.resnet50(norm_layer=norm)
    else:
        raise NotImplementedError(f"Unsupported ResNet size: {size}")

    if weights is not None:
        state_dict = weight_enum.verify(weights).get_state_dict(progress=True)
        if norm is not nn.BatchNorm2d:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if "running_mean" not in k and "running_var" not in k
            }
        backbone.load_state_dict(state_dict)
    return backbone


class ResNetEncoder(nn.Module):
    """
    ResNet image encoder (dit-policy compatible).

    Expects x of shape (B, C, H, W). With avg_pool=True (default), returns
    (B, 1, embed_dim). With avg_pool=False, returns (B, n_tokens, embed_dim).
    """

    def __init__(
        self,
        size: int = 18,
        norm_cfg: Optional[Dict] = None,
        weights: Optional[str] = "IMAGENET1K_V1",
        restore_path: str = "",
        avg_pool: bool = True,
        conv_repeat: int = 1,
    ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(name="group_norm", num_groups=16)

        norm_layer = _make_norm(norm_cfg)
        backbone = _construct_resnet(size, norm_layer, weights)
        backbone.fc = nn.Identity()
        if not avg_pool:
            backbone.avgpool = nn.Identity()

        if conv_repeat > 1:
            w = backbone.conv1.weight.data.repeat((1, conv_repeat, 1, 1))
            backbone.conv1.weight.data = w
            backbone.conv1.in_channels *= conv_repeat

        self._model = backbone
        self._size = size
        self._avg_pool = avg_pool

        if restore_path:
            self._load_restore_path(restore_path)

    def _load_restore_path(self, restore_path: str) -> None:
        print(f"Restoring ResNetEncoder from {restore_path}")
        state_dict = torch.load(restore_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "features" in state_dict:
                state_dict = state_dict["features"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._avg_pool:
            return self._model(x)[:, None]
        B = x.shape[0]
        x = self._model(x)
        x = x.reshape((B, self.embed_dim, -1))
        return x.transpose(1, 2)

    @property
    def embed_dim(self) -> int:
        return {18: 512, 34: 512, 50: 2048}[self._size]

    @property
    def n_tokens(self) -> int:
        if self._avg_pool:
            return 1
        return 49  # 224x224 input


class ImageEncoder(nn.Module):
    """Single-frame image encoder: ResNet -> MLP -> (B, output_dim)."""

    def __init__(
        self,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        resnet_size: int = 18,
        image_weights: Optional[str] = "IMAGENET1K_V1",
    ):
        super().__init__()
        self.resnet = ResNetEncoder(
            size=resnet_size,
            weights=image_weights,
            avg_pool=True,
        )
        self.proj = MLP(
            input_dim=self.resnet.embed_dim,
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) single frame."""
        h = self.resnet(x)
        if h.dim() == 3:
            h = h.squeeze(1)
        return self.proj(h)


class TaskEncoder(nn.Module):
    """
    LIBERO-style task conditioning via one-hot task ID.

    Accepts task indices (B,) or one-hot vectors (B, num_tasks).
    Returns an MLP embedding of shape (B, output_dim).
    """

    def __init__(
        self,
        num_tasks: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.mlp = MLP(
            input_dim=num_tasks,
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=output_dim,
        )

    def to_one_hot(self, task: torch.Tensor) -> torch.Tensor:
        if task.dim() == 1:
            return torch.nn.functional.one_hot(
                task.long(), num_classes=self.num_tasks
            ).float()
        if task.shape[-1] == self.num_tasks:
            return task.float()
        raise ValueError(
            f"task must be (B,) indices or (B, {self.num_tasks}) one-hot, "
            f"got shape {tuple(task.shape)}"
        )

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.to_one_hot(task))


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GaussianHead(nn.Module):
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(input_dim, z_dim)
        self.logvar = nn.Linear(input_dim, z_dim)

    def forward(self, h: torch.Tensor):
        return self.mu(h), self.logvar(h)
