import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

def calculate_psnr(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate PSNR between predicted and clean images.

    Args:
        pred (torch.Tensor): Predicted images of shape (B, C, H, W).
        clean (torch.Tensor): Ground truth images of shape (B, C, H, W).
        max_val (float): Maximum possible pixel value (e.g., 1.0 for normalized, 255 for 8-bit).

    Returns:
        torch.Tensor: PSNR value averaged over the batch.
    """
    mse = F.mse_loss(pred, clean, reduction='none')
    mse_per_image = mse.view(mse.shape[0], -1).mean(dim=1)  # Mean per image in batch

    # Avoid division by zero
    psnr = 10 * torch.log10((max_val ** 2) / (mse_per_image + 1e-8))

    return psnr.mean()  # Return average PSNR over the batch


def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(0)
    window_2d = (window_1d.T @ window_1d).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(pred: torch.Tensor, clean: torch.Tensor, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate SSIM between predicted and clean images.

    Args:
        pred (torch.Tensor): Predicted images of shape (B, C, H, W).
        clean (torch.Tensor): Ground truth images of shape (B, C, H, W).
        window_size (int): Size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian window.
        max_val (float): Maximum possible pixel value (e.g., 1.0 for normalized, 255 for 8-bit).

    Returns:
        torch.Tensor: SSIM value averaged over the batch.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    _, channels, _, _ = pred.size()
    window = gaussian_window(window_size, sigma, channels).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(clean, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(clean * clean, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * clean, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_mse(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error (MSE) between predicted and clean images.

    Args:
        pred (torch.Tensor): Predicted images of shape (B, C, H, W).
        clean (torch.Tensor): Ground truth images of shape (B, C, H, W).

    Returns:
        torch.Tensor: MSE value averaged over the batch.
    """
    return torch.mean((pred - clean) ** 2)


def plot_metrics_against_sigma(df):
    """
    Plots PSNR, SSIM, and MSE against sigma in vertically stacked subplots.

    Args:
        df (pd.DataFrame): DataFrame with columns 'sigma', 'PSNR', 'SSIM', and 'MSE'.
    """
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    # Plot PSNR
    axs[0].plot(df['sigma'], df['PSNR'], marker='o')
    axs[0].set_ylabel('PSNR (dB)')
    axs[0].set_title('PSNR vs Sigma')
    axs[0].grid(True)

    # Plot SSIM
    axs[1].plot(df['sigma'], df['SSIM'], marker='o')
    axs[1].set_ylabel('SSIM')
    axs[1].set_title('SSIM vs Sigma')
    axs[1].grid(True)

    # Plot MSE
    axs[2].plot(df['sigma'], df['MSE'], marker='o')
    axs[2].set_ylabel('MSE')
    axs[2].set_title('MSE vs Sigma')
    axs[2].set_xlabel('Sigma')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()