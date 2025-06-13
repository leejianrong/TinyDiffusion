import torch
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_noise_level(t, T):
    # map [0, T] to [-6, +6]
    x = (t / T) * 12.0 - 6.0
    return 1.0 / (1.0 + torch.exp(-x)) # range (0, 1)
    

def cosine_noise_level(t, T, s=0.008):
    """
    Modified cosine noise schedule: sigma(0) = 0

    Args:
        t (int, float, or torch.Tensor): Current timestep(s).
        T (int): Total number of timesteps.
        s (float): Small offset parameter (default 0.008).

    Returns:
        torch.Tensor: Noise level at timestep t.
    """
    # Ensure tensor input
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    else:
        t = t.clone().detach().float()

    T = float(T)

    # Normalized time
    f_t = (t / T + s) / (1 + s)

    # Normalized ᾱ with ᾱ(0) = 1 and ᾱ(T) = 0
    alpha_bar_t = (torch.cos(f_t * torch.pi / 2) ** 2) / (torch.cos(torch.tensor(s / (1 + s) * torch.pi / 2)) ** 2)

    # Clamp to [0, 1] to prevent numerical issues
    alpha_bar_t = torch.clamp(alpha_bar_t, 0.0, 1.0)

    return torch.sqrt(1 - alpha_bar_t)


def img_to_np(x: torch.Tensor):
    x = x.detach().cpu().numpy().transpose(1, 2, 0)
    return (x * 255).astype(np.uint8)


# Alternative version if you want to add row labels on the left
def plot_noisy_clean_images_with_labels(noisy_images, clean_images, timesteps, n_cols=10):
    """
    Plot two rows of images with row labels on the left side.
    """
    # Sort images by timestep in ascending order
    timestep_values = [t.item() if hasattr(t, 'item') else t for t in timesteps]
    sorted_indices = sorted(range(len(timestep_values)), key=lambda i: timestep_values[i])
    
    # Reorder all lists according to sorted timesteps
    noisy_images = [noisy_images[i] for i in sorted_indices]
    clean_images = [clean_images[i] for i in sorted_indices]
    timesteps = [timesteps[i] for i in sorted_indices]
    
    n_images = min(len(noisy_images), len(clean_images), n_cols)
    
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_images):
        noisy_np = img_to_np(noisy_images[i])
        clean_np = img_to_np(clean_images[i])
        
        # Check channels for grayscale
        cmap_noisy = 'gray' if noisy_np.shape[-1] == 1 or len(noisy_np.shape) == 2 else None
        cmap_clean = 'gray' if clean_np.shape[-1] == 1 or len(clean_np.shape) == 2 else None
        
        # Plot noisy image
        axes[0, i].imshow(noisy_np, cmap=cmap_noisy)
        axes[0, i].set_title(f"t = {timesteps[i].item() if hasattr(timesteps[i], 'item') else timesteps[i]}", 
                            fontsize=10)
        axes[0, i].axis("off")
        
        # Plot clean image
        axes[1, i].imshow(clean_np, cmap=cmap_clean)
        axes[1, i].axis("off")
        
        # Add row labels on the leftmost column
        if i == 0:
            axes[0, i].set_ylabel("Noisy", rotation=90, fontsize=12, labelpad=20)
            axes[1, i].set_ylabel("Clean", rotation=90, fontsize=12, labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal spacing between columns
    plt.show()


