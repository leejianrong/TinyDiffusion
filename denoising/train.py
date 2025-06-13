import torch
import time
from tqdm import tqdm

import torch
import time
from .loss import LiftingDenoiserLoss


def print_loss_components(loss_components, end=", "):
    print("\t", end="")
    for name, loss_val in loss_components.items():
        if loss_val > 0.0:
            print(f"{name}: {loss_val:.4f}", end=end) 
    print("")

def train_denoiser(
    model,
    device,
    crit,
    opt,
    train_loader,
    val_loader,
    num_epochs,
    is_custom_loss_fn = False,
):
    train_losses = []
    val_losses = []
        
    for epoch in range(1, num_epochs + 1):

        tic = time.time()

        # ---- TRAIN -------------------------------------------------------
        model.train()
        running_loss = 0.0

        for noisy, t, clean in train_loader:
            noisy, t, clean = noisy.to(device), t.to(device), clean.to(device)
            # print(t.shape)
            pred  = model(noisy, t)       # expected signature: f(x, t) -> denoised
            
            if is_custom_loss_fn:
                loss, loss_components = crit(pred, clean, model)
            else:
                loss = crit(pred, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * noisy.size(0)

        train_mse = running_loss / len(train_loader.dataset)
        train_losses.append(train_mse)

        # ---- VALIDATE ----------------------------------------------------
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for noisy, t, clean in val_loader:
                noisy, t, clean = noisy.to(device), t.to(device), clean.to(device)
                pred  = model(noisy, t)
                if is_custom_loss_fn:
                    loss, loss_components = crit(pred, clean, model)
                else:
                    loss = crit(pred, clean)
                val_running += loss.item() * noisy.size(0)

        val_mse = val_running / len(val_loader.dataset)
        val_losses.append(val_mse)
        epoch_time = time.time() - tic

        # ---- LOG ---------------------------------------------------------
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_mse:.6f} | "
            f"Val Loss: {val_mse:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        if is_custom_loss_fn:
            print_loss_components(loss_components)

    return train_losses, val_losses