import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from .utils import sigmoid_noise_level

class NoisyImageDataset(Dataset):
    """
    Supports CIFAR-10 (RGB 32x32) and Fashion-MNIST (grey 28x28).
    Args
    ----
    dataset        : "CIFAR10" | "FashionMNIST"
    replicate_chan : If True and dataset is grayscale, repeat the single
                     channel 3x so models written for RGB still work.
    T              : Number of diffusion / noise steps
    """
    DATASET_CHOICES = {
        "CIFAR10": datasets.CIFAR10,
        "FashionMNIST": datasets.FashionMNIST,
    }

    def __init__(
        self,
        dataset="CIFAR10",
        root="./data",
        train=True,
        download=True,
        replicate_chan=False,
        T=1_000,
    ):
        if dataset not in self.DATASET_CHOICES:
            raise ValueError(f"Dataset must be one of {list}")

        tf_chain = []

        # resize FashionMNIST from 28x28 to 32x32 so every image is 32x32
        if dataset == "FashionMNIST":
            tf_chain.append(
                transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR)
            )

        tf_chain.append(transforms.ToTensor())

        # optional channel replication for grayscale data
        if replicate_chan and dataset == "FashionMNIST":
            tf_chain.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            
        self.base_tf = transforms.Compose(tf_chain)

        self.clean_set = self.DATASET_CHOICES[dataset](
            root=root, train=train, download=download, transform=self.base_tf
        )
        self.T = T

    def __len__(self):
        return len(self.clean_set)

    def __getitem__(self, idx):
        clean_img, _ = self.clean_set[idx]
        t = torch.randint(0, self.T, (1,), dtype=torch.long) # scalar
        sigma = sigmoid_noise_level(t.float(), self.T).sqrt()
        noisy = torch.clamp(
            clean_img + torch.randn_like(clean_img) * sigma, 
            0.0,
            1.0,
        )
        return noisy, t.squeeze(0), clean_img






















        