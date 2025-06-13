import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Dict

def is_valid_weight(weight: float):
    return (weight is not None) and (weight > 0.0)


def conv2d_spectral_norm(w: Tensor) -> Tensor:
    """
    w: (C_out, C_in, k, k)  -> reshape to  (C_out, C_in * k * k)
    Returns the L2 spectral norm (largest singular value).
    """
    w_mat = w.flatten(1)                                # shape (C_out, N)
    # torch.linalg.svdvals is differentiable and needs only one line:
    return torch.linalg.svdvals(w_mat)[0]               # top singular value


def spectral_norm_loss(module: nn.Module,
                       target: float = 1.0,
                       coeff:  float = 1.0) -> Tensor:
    """
    Collects every nn.Conv2d inside `module`, computes its spectral
    norm, and builds the penalty  Σ (‖W‖₂ - target)².
    """
    loss = 0.0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            sigma = conv2d_spectral_norm(m.weight)
            loss += (sigma - target).pow(2)
    return coeff * loss


class LiftingDenoiserLoss(nn.Module):
    """Custom loss that combines reconstruction loss with dictionary orthogonality constraint."""
    
    def __init__(
        self, 
        reconstruction_weight: float = 1.0, 
        splitmerge_orthogonal_weight: float | None = 1e-4,
        clista_orthogonal_weight: float | None = 1e-4, 
        spectral_norm_weight: float | None = 1e-4,
    ):
        """
        Args:
            reconstruction_weight: Weight for the main reconstruction loss (MSE)
            orthogonal_weight: Weight for the orthogonality constraint
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.splitmerge_orthogonal_weight = splitmerge_orthogonal_weight
        self.clista_orthogonal_weight = clista_orthogonal_weight
        self.spectral_norm_weight = spectral_norm_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        predictions, 
        targets, 
        model
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined loss with reconstruction and orthogonality terms.
        
        Args:
            predictions: Model output (denoised image)
            targets: Ground truth clean image
            model: The CLISTA model to extract dictionaries from
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # compute reconstruction loss (main objective)
        reconstruction_loss = self.mse_loss(predictions, targets)

        
        # compute custom losses
        custom_loss_components = {}
        
        if hasattr(model, "detail_denoiser_type") and \
            model.detail_denoiser_type == "clista" and \
            is_valid_weight(self.clista_orthogonal_weight):
            
            try:
                custom_loss_components["clista_orthogonal"] = model.detail_denoiser.orthogonality_loss()
            except ValueError as e:
                print(f"Warning: {e}")
            
        if hasattr(model, "split_merge_type") and \
            model.split_merge_type == "learnable" and \
            is_valid_weight(self.spectral_norm_weight):
            
            try:
                custom_loss_components["splitmerge_orthogonal"] = model.split_merge.orthogonality_loss()
            except ValueError as e:
                print(f"Warning: {e}")
            
        if hasattr(model, "lifting_steps") and \
            is_valid_weight(self.spectral_norm_weight):
            total_spectral_norm_loss = 0.0
            for lifting_step in model.lifting_steps:
                 total_spectral_norm_loss += spectral_norm_loss(lifting_step)
                
            custom_loss_components["lifting_spectral_norm"] = total_spectral_norm_loss / len(model.lifting_steps)
                
        # get weights sum of losses
        total_loss = self.reconstruction_weight * reconstruction_loss
        
        if is_valid_weight(self.clista_orthogonal_weight):
            total_loss += self.clista_orthogonal_weight * custom_loss_components["clista_orthogonal"]
            
        if is_valid_weight(self.splitmerge_orthogonal_weight):
            total_loss += self.splitmerge_orthogonal_weight * custom_loss_components["splitmerge_orthogonal"]

        if is_valid_weight(self.spectral_norm_weight):
            total_loss += self.spectral_norm_weight * custom_loss_components["lifting_spectral_norm"]
        
        # return loss components for monitoring
        for name, val in custom_loss_components.items():
            custom_loss_components[name] = val.item() if isinstance(val, Tensor) else val
            
        loss_components = {
            'reconstruction_loss': reconstruction_loss.item(),
            'total_loss': total_loss.item()
        }
        loss_components.update(custom_loss_components)
        return total_loss, loss_components

