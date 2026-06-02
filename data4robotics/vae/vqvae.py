from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ImageEncoder, MLP, TaskEncoder

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
        self.proj = nn.Linear(feature_dim, z_dim)

    def forward(
        self,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> torch.Tensor:
        bsz = actions.shape[0]
        feats = [
            self.action_mlp(
                actions.reshape(bsz, self.action_chunk_size * self.action_dim)
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
        return self.proj(self.fusion(h))


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
        bsz = z.shape[0]
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
        return recon.reshape(bsz, self.action_chunk_size, self.action_dim)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, z_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.z_dim = z_dim
        self.embedding = nn.Embedding(codebook_size, z_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_e @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, z_q, indices, z_e


class ActionVQVAE(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        num_layers: int = 3,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        z_dim: int = 16,
        codebook_size: int = 512,
        commitment_beta: float = 0.25,
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
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
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
        self.quantizer = VectorQuantizer(codebook_size=codebook_size, z_dim=z_dim)
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

    def encode(
        self,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(actions, proprio, task, images)
        z_q_st, _, _, _ = self.quantizer(z_e)
        return z_q_st, torch.zeros_like(z_q_st)

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
        del deterministic
        z_e = self.encoder(actions, proprio, task, images)
        z_q_st, z_q, indices, _ = self.quantizer(z_e)
        recon = self.decode(z_q_st, proprio, task, images)
        return {
            "recon": recon,
            "z": z_q_st,
            "mu": z_q_st,
            "logvar": torch.zeros_like(z_q_st),
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
        }

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        proprio: Optional[torch.Tensor] = None,
        task: TaskInput = None,
        images: ImagesInput = None,
    ) -> torch.Tensor:
        indices = torch.randint(
            low=0, high=self.codebook_size, size=(batch_size,), device=device
        )
        z = self.quantizer.embedding(indices)
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


def action_vqvae_loss(
    outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    commitment_beta: float = 0.25,
) -> Dict[str, torch.Tensor]:
    recon_loss = masked_mse_loss(outputs["recon"], actions, mask)
    codebook_loss = F.mse_loss(outputs["z_q"], outputs["z_e"].detach())
    commitment_loss = F.mse_loss(outputs["z_e"], outputs["z_q"].detach())
    loss = recon_loss + codebook_loss + commitment_beta * commitment_loss
    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "codebook_loss": codebook_loss,
        "commitment_loss": commitment_loss,
    }

