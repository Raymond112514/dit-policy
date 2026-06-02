from .vae import ActionVAE, action_vae_loss, masked_mse_loss, kl_divergence_standard_normal
from .vqvae import ActionVQVAE, action_vqvae_loss
from .networks import MLP, ImageEncoder, TaskEncoder, GaussianHead, ResNetEncoder
from .preprocess import get_vae_image_transform


def vae_encode_device() -> str:
    """Device for one-time VAE pre-encoding (cuda if available, else cpu)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_action_vae(checkpoint_path: str, device=None):
    """Load an ActionVAE from a train_vae.py checkpoint.

    Reconstructs the model architecture from the saved args dict and loads weights.
    Returns the model in eval mode on the requested device.
    """
    import torch

    if device is None:
        device = vae_encode_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    encoder_use_proprio = args.get("encoder_use_proprio", False)
    decoder_use_proprio = args.get("decoder_use_proprio", False)
    encoder_use_task = args.get("encoder_use_task", False)
    decoder_use_task = args.get("decoder_use_task", False)
    encoder_use_image = args.get("encoder_use_image", False)
    decoder_use_image = args.get("decoder_use_image", False)

    proprio_dim = args.get("proprio_dim", 9) if (encoder_use_proprio or decoder_use_proprio) else None
    num_tasks = args.get("num_tasks", 90) if (encoder_use_task or decoder_use_task) else None

    model_type = str(args.get("model_type", "action_vae")).lower()
    common_kwargs = dict(
        action_dim=args["action_dim"],
        action_chunk_size=args["action_chunk_size"],
        num_layers=args.get("num_layers", 3),
        hidden_dim=args.get("hidden_dim", 512),
        feature_dim=args.get("feature_dim", 256),
        z_dim=args.get("z_dim", 16),
        encoder_use_proprio=encoder_use_proprio,
        decoder_use_proprio=decoder_use_proprio,
        proprio_dim=proprio_dim,
        encoder_use_task=encoder_use_task,
        decoder_use_task=decoder_use_task,
        num_tasks=num_tasks,
        encoder_use_image=encoder_use_image,
        decoder_use_image=decoder_use_image,
        resnet_size=args.get("resnet_size", 18),
    )
    if model_type in ("action_vqvae", "vqvae", "action-vqvae"):
        model = ActionVQVAE(
            codebook_size=args.get("codebook_size", 512),
            commitment_beta=args.get("commitment_beta", 0.25),
            **common_kwargs,
        )
    else:
        model = ActionVAE(**common_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)
