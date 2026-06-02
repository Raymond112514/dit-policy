from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .networks import GaussianHead, ImageEncoder, MLP, TaskEncoder

ImagesInput = Optional[torch.Tensor]
TaskInput = Union[torch.Tensor, None]


def _build_image_encoder(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    resnet_size: int,
) -> ImageEncoder:
    return ImageEncoder(
        output_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        resnet_size=resnet_size,
    )


class Encoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        num_layers: int,
        hidden_dim: int,
        feature_dim: int,
        z_dim: int,
        use_proprio: bool = False,
        proprio_dim: Optional[int] = None,
        use_task: bool = False,
        num_tasks: Optional[int] = None,
        use_image: bool = False,
        resnet_size: int = 18,
        image_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.use_task = use_task
        self.use_image = use_image
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        self.action_mlp = MLP(
            input_dim=action_dim * action_chunk_size,
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=feature_dim,
        )

        self.proprio_mlp = None
        if use_proprio:
            self.proprio_mlp = MLP(
                input_dim=proprio_dim,
                hidden_dims=[hidden_dim] * num_layers,
                output_dim=feature_dim,
            )

        self.task_encoder = None
        if use_task:
            if num_tasks is None:
                raise ValueError("num_tasks required when use_task=True")
            self.task_encoder = TaskEncoder(
                num_tasks=num_tasks,
                output_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

        self.image_encoder = None
        if use_image:
            self.image_encoder = image_encoder or _build_image_encoder(
                feature_dim, hidden_dim, num_layers, resnet_size
            )

        fusion_input_dim = feature_dim * (
            1 + int(use_proprio) + int(use_task) + int(use_image)
        )
        self.fusion = MLP(
            input_dim=fusion_input_dim,
            hidden_dims=[hidden_dim],
            output_dim=feature_dim,
        )
        self.gaussian_head = GaussianHead(input_dim=feature_dim, z_dim=z_dim)

    def forward(
        self,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ):
        B = actions.shape[0]
        feats = [
            self.action_mlp(
                actions.reshape(B, self.action_chunk_size * self.action_dim)
            )
        ]
        if self.use_proprio:
            assert proprio is not None
            feats.append(self.proprio_mlp(proprio))
        if self.use_task:
            assert task is not None
            feats.append(self.task_encoder(task))
        if self.use_image:
            assert images is not None
            feats.append(self.image_encoder(images))

        h = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        h = self.fusion(h)
        return self.gaussian_head(h)


class Decoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        num_layers: int,
        hidden_dim: int,
        feature_dim: int,
        z_dim: int,
        use_proprio: bool = False,
        proprio_dim: Optional[int] = None,
        use_task: bool = False,
        num_tasks: Optional[int] = None,
        use_image: bool = False,
        image_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.use_proprio = use_proprio
        self.use_task = use_task
        self.use_image = use_image

        decoder_input_dim = z_dim
        self.proprio_mlp = None
        if use_proprio:
            self.proprio_mlp = MLP(
                input_dim=proprio_dim,
                hidden_dims=[hidden_dim] * num_layers,
                output_dim=feature_dim,
            )
            decoder_input_dim += feature_dim

        self.task_encoder = None
        if use_task:
            if num_tasks is None:
                raise ValueError("num_tasks required when use_task=True")
            self.task_encoder = TaskEncoder(
                num_tasks=num_tasks,
                output_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
            decoder_input_dim += feature_dim

        self.image_encoder = None
        if use_image:
            self.image_encoder = image_encoder
            decoder_input_dim += feature_dim

        self.mlp = MLP(
            input_dim=decoder_input_dim,
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=action_chunk_size * action_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> torch.Tensor:
        B = z.shape[0]
        parts = [z]
        if self.use_proprio:
            assert proprio is not None
            parts.append(self.proprio_mlp(proprio))
        if self.use_task:
            assert task is not None
            parts.append(self.task_encoder(task))
        if self.use_image:
            assert images is not None and self.image_encoder is not None
            parts.append(self.image_encoder(images))

        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        recon = self.mlp(h)
        return recon.reshape(B, self.action_chunk_size, self.action_dim)


class ActionVAE(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        num_layers: int = 3,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        z_dim: int = 16,
        encoder_use_proprio: bool = False,
        decoder_use_proprio: bool = False,
        proprio_dim: Optional[int] = None,
        encoder_use_task: bool = False,
        decoder_use_task: bool = False,
        num_tasks: Optional[int] = None,
        encoder_use_image: bool = False,
        decoder_use_image: bool = False,
        resnet_size: int = 18,
    ):
        super().__init__()

        if (encoder_use_proprio or decoder_use_proprio) and proprio_dim is None:
            raise ValueError("proprio_dim required when encoder/decoder uses proprio.")
        if (encoder_use_task or decoder_use_task) and num_tasks is None:
            raise ValueError("num_tasks required when encoder/decoder uses task.")

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.z_dim = z_dim
        self.num_tasks = num_tasks
        self.encoder_use_proprio = encoder_use_proprio
        self.decoder_use_proprio = decoder_use_proprio
        self.encoder_use_task = encoder_use_task
        self.decoder_use_task = decoder_use_task
        self.encoder_use_image = encoder_use_image
        self.decoder_use_image = decoder_use_image

        shared_image_enc = None
        if encoder_use_image or decoder_use_image:
            shared_image_enc = _build_image_encoder(
                feature_dim, hidden_dim, num_layers, resnet_size
            )

        enc_image = shared_image_enc if encoder_use_image else None
        dec_image = shared_image_enc if decoder_use_image else None
        if decoder_use_image and not encoder_use_image:
            dec_image = _build_image_encoder(
                feature_dim, hidden_dim, num_layers, resnet_size
            )

        self.encoder = Encoder(
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            z_dim=z_dim,
            use_proprio=encoder_use_proprio,
            proprio_dim=proprio_dim,
            use_task=encoder_use_task,
            num_tasks=num_tasks,
            use_image=encoder_use_image,
            resnet_size=resnet_size,
            image_encoder=enc_image,
        )
        self.decoder = Decoder(
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            z_dim=z_dim,
            use_proprio=decoder_use_proprio,
            proprio_dim=proprio_dim,
            use_task=decoder_use_task,
            num_tasks=num_tasks,
            use_image=decoder_use_image,
            image_encoder=dec_image,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ):
        return self.encoder(actions, proprio, task, images)

    def decode(
        self,
        z: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> torch.Tensor:
        return self.decoder(z, proprio, task, images)

    def forward(
        self,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(actions, proprio, task, images)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon = self.decode(z, proprio, task, images)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> torch.Tensor:
        z = torch.randn(batch_size, self.z_dim, device=device)
        return self.decode(z, proprio, task, images)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    mse = (pred - target) ** 2
    if mask is None:
        return mse.mean()
    mse = mse * mask[..., None]
    denom = mask.sum() * pred.shape[-1] + eps
    return mse.sum() / denom


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim.sum(dim=-1).mean()


def action_vae_loss(
    outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    beta: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    recon_loss = masked_mse_loss(outputs["recon"], actions, mask)
    kl_loss = kl_divergence_standard_normal(outputs["mu"], outputs["logvar"])
    loss = recon_loss + beta * kl_loss
    return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
