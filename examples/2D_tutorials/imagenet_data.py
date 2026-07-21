import torch
# from datasets import load_dataset
# from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 as transforms


def get_transforms(dataset: str):
    if dataset == "cifar10":
        size = 32
    else:
        size = 256

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


class MyDataset(Dataset):
    """
    Dataset class for ImageNet and CIFAR-10 with optional VAE compression.

    Args:
        dataset: Dataset name ("imagenet" or "cifar10")
        split: Dataset split ("train" or "validation")
        use_vae_compression: If True, compress images using Stable Diffusion VAE
        device: Device to run VAE on ("cuda" or "cpu")
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        split: str = "train",
        use_vae_compression: bool = False,
        device: str = "cuda",
    ):
        assert split in ["train", "validation"], "Split must be either train or validation"
        self.dataset_name = dataset
        self.use_vae_compression = use_vae_compression
        self.device = device

        if dataset == "imagenet":
            self.dataset = load_dataset("benjamin-paine/imagenet-1k-256x256")[split]
        elif dataset == "cifar10":
            self.dataset = CIFAR10(root=data_path, train=(split == "train"), download=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.transforms = get_transforms(dataset)

        # Initialize VAE if compression is enabled
        # VAE compression reduces image dimensions from (3, H, W) to (4, H/8, W/8)
        if self.use_vae_compression:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
            self.vae.eval()

    def __len__(self):
        return len(self.dataset)

    def _compress_with_vae(self, image: torch.Tensor) -> torch.Tensor:
        """Compress image using VAE encoder"""
        with torch.no_grad():
            # Ensure image is in the right format for VAE (batch dimension, channels first)
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension

            # Move to device
            image = image.to(self.device)

            # Encode with VAE
            latent = self.vae.encode(image).latent_dist.sample()

            # Scale by VAE scaling factor
            latent = latent * self.vae.config.scaling_factor

            return latent.squeeze(0)  # Remove batch dimension

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.dataset_name == "imagenet":
            image = item["image"]
            label = item["label"]
        elif self.dataset_name == "cifar10":
            image = item[0]
            label = item[1]
        image = self.transforms(image)

        # Apply VAE compression if enabled
        if self.use_vae_compression:
            image = self._compress_with_vae(image)

        return image, label

    def sample(self, bs, class_idx=None):
        # do the sample without replacement
        perm = torch.randperm(len(self.dataset))
        indices = perm[:bs]

        # indices = torch.randint(0, len(self.dataset), (bs,))

        datum = []
        # breakpoint()
        for idx in indices:
            idx = idx.item()
            try:
                item = self.dataset[idx]
                if self.dataset_name == "imagenet":
                    image = item["image"]
                elif self.dataset_name == "cifar10":
                    image = item[0]
                image = self.transforms(image)

                # Apply VAE compression if enabled
                if self.use_vae_compression:
                    image = self._compress_with_vae(image)

                datum.append(image)
                datum.append(image)
            except Exception as e:
                print(f"Error occurred while processing index {idx}: {e}")
                continue
        datum = torch.stack(datum)
        return datum
