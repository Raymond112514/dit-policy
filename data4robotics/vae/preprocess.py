"""VAE image preprocessing — must match latent/data/transforms.py exactly.

Used both during dataset pre-encoding (replay_buffer._encode_with_vae) and
at deployment time when the VAE decoder is called during rollout.
"""

from torchvision import transforms


def get_vae_image_transform(size: int = 128):
    """Return the same Resize + ImageNet-normalize pipeline used in VAE training.

    Matches latent/data/transforms.py::get_image_transform("preproc", size).
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size), antialias=False),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
