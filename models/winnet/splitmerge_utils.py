import torch
import torch.nn.functional as F

def match_spatial_dims(x, x_desired):
    """
    Resize tensor x to match the spatial dimensions of x_desired.
    
    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W)
        x_desired (torch.Tensor): Reference tensor of shape (N, C, H_desired, W_desired)
    
    Returns:
        torch.Tensor: Resized tensor with spatial dims matching x_desired
    """
    # Use .size() consistently instead of mixing .shape and .size()
    desired_height = x_desired.size(2)
    desired_width = x_desired.size(3)
    
    current_height = x.size(2)
    current_width = x.size(3)
    
    if current_height != desired_height or current_width != desired_width:
        x = F.interpolate(
            x, 
            size=(desired_height, desired_width),
            mode='bilinear', 
            align_corners=False
        )
    return x


def orthonormalize_groupwise(w):
    """
    Enforce (approximate) orthonormality inside every 'groups' block.
    w : (out_ch, 1, kH, kW) where out_ch = groups * K
    """
    groups, K, kH, kW = w.shape[0] // w.groups, w.shape[0] // w.groups, *w.shape[2:]
    w_reshaped = w.view(groups, K, -1)                       # (G, K, kH*kW)
    # Gram–Schmidt lite: normalise each vector
    w_reshaped = F.normalize(w_reshaped, dim=2)
    return w_reshaped.view_as(w)