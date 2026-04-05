"""
Training script for Complex-Valued PhaseNet

This script trains the complex-valued PhaseNet for video frame interpolation.
It processes video frames through complex steerable pyramids and trains the
network to predict interpolated frames.
"""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import Triplets, show_Triplets_batch
from net.complex_phasenet import ComplexPhaseNet, ComplexTotalLoss, complex_input_convert


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the complex-valued PhaseNet model."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--feature-dim", type=int, default=32, help="Model feature dimension.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional DAVIS-style dataset path override.",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="If > 0, train on only the first N triplets for debugging/overfit checks.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="If > 0, stop each epoch after this many optimizer steps.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log loss components every N steps.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="Save debug images every N steps.",
    )
    parser.add_argument(
        "--debug-save-dir",
        type=Path,
        default=Path("./debug_images"),
        help="Directory for saving debug images.",
    )
    # Loss weights (passed directly to ComplexTotalLoss)
    parser.add_argument("--img-weight", type=float, default=3.0)
    parser.add_argument("--residual-weight", type=float, default=2.0)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--amp-weight", type=float, default=1.2)
    parser.add_argument("--amp-imag-loss-weight", type=float, default=0.05)
    parser.add_argument("--phase-unit-weight", type=float, default=0.01)
    parser.add_argument("--grad-weight", type=float, default=2.0)
    return parser.parse_args()


def resolve_dataset_path():
    """Resolve the DAVIS dataset path from env/configured/common locations."""
    candidate_paths = []

    env_path = os.environ.get('PHASENET_DATASET_PATH')
    if env_path:
        candidate_paths.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parent
    candidate_paths.extend([
        Path('/home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p'),
        Path('/home/Salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p'),
        repo_root / 'DAVIS-data' / 'DAVIS' / 'JPEGImages' / '480p',
    ])

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return str(candidate_paths[0] if candidate_paths else repo_root)


def extract_complex_coefficients(pyr_coeff):
    """
    Extract real and imaginary parts from pyramid coefficients.
    
    Args:
        pyr_coeff: List of pyramid coefficients from SCFpyr_PyTorch
                   [high_pass, [bands...], ..., low_pass]
    
    Returns:
        tuple: (coeff_real, coeff_imag) - lists of real and imaginary parts
    """
    coeff_real = []
    coeff_imag = []
    
    for level in pyr_coeff:
        if isinstance(level, list):
            # This is a band level with orientations
            level_real = []
            level_imag = []
            for band in level:
                # band is complex tensor [..., 2] where last dim is [real, imag]
                if band.dim() == 4 and band.shape[-1] == 2:
                    level_real.append(band[..., 0])  # Real part
                    level_imag.append(band[..., 1])  # Imaginary part
                else:
                    # If not complex, assume real only
                    level_real.append(band)
                    level_imag.append(torch.zeros_like(band))
            coeff_real.append(level_real)
            coeff_imag.append(level_imag)
        else:
            # This is high-pass or low-pass (real valued)
            coeff_real.append(level)
            coeff_imag.append(torch.zeros_like(level))
    
    return coeff_real, coeff_imag


def get_complex_input(batch_coeff_list):
    """
    Convert batch of triplet coefficients to training format.

    Returns:
        tuple:
          train_real, train_imag, truth_real, truth_imag,
          highpass_start, highpass_end
    """
    train_real_batch = []
    train_imag_batch = []
    truth_real_batch = []
    truth_imag_batch = []
    highpass_start_batch = []
    highpass_end_batch = []

    for triplet_coeff in batch_coeff_list:
        tri_real, tri_imag = extract_complex_coefficients(triplet_coeff)
        tr_r, tr_i, truth_r, truth_i, hp_s, hp_e = complex_input_convert(tri_real, tri_imag)

        train_real_batch.append(tr_r)
        train_imag_batch.append(tr_i)
        truth_real_batch.append(truth_r)
        truth_imag_batch.append(truth_i)
        highpass_start_batch.append(hp_s)
        highpass_end_batch.append(hp_e)

    train_real = []
    train_imag = []
    truth_real = []
    truth_imag = []

    for level_idx in range(len(train_real_batch[0])):
        train_real.append(torch.stack([item[level_idx] for item in train_real_batch]))
        train_imag.append(torch.stack([item[level_idx] for item in train_imag_batch]))
        truth_real.append(torch.stack([item[level_idx] for item in truth_real_batch]))
        truth_imag.append(torch.stack([item[level_idx] for item in truth_imag_batch]))

    highpass_start = torch.stack(highpass_start_batch)
    highpass_end = torch.stack(highpass_end_batch)

    return train_real, train_imag, truth_real, truth_imag, highpass_start, highpass_end

def output_convert_complex(pred_real, pred_imag, highpass=None):
    """
    Convert complex-valued predictions back to pyramid coefficient format.
    
    Args:
        pred_real, pred_imag: Lists of predicted complex coefficients
    
    Returns:
        List of coefficients in pyramid format
    """
    coeff = []
    
    # Residual level
    coeff.append(pred_real[0].squeeze(1))
    
    # Band levels
    for i in range(1, len(pred_real)):
        bands_num = pred_real[i].shape[1] // 2
        band = []
        
        for j in range(bands_num):
            # Get amplitude and phase
            amp_r = pred_real[i][:, j, :, :]
            amp_i = pred_imag[i][:, j, :, :]
            phase_r = pred_real[i][:, j + bands_num, :, :]
            phase_i = pred_imag[i][:, j + bands_num, :, :]
            
            # Convert amplitude (treat as real) and phase (as complex angle)
            amp = torch.sqrt(amp_r**2 + amp_i**2 + 1e-8)
            phase = torch.atan2(phase_i + 1e-8, phase_r + 1e-8)
            
            # Reconstruct complex coefficient
            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            
            # Stack as complex tensor
            complex_coeff = torch.stack([real, imag], -1)
            band.append(complex_coeff)
        
        coeff.insert(0, band)
    
    # Add high-pass (zeros) on the same device/dtype as the predictions.
    if highpass is None:
        highpass = pred_real[-1].new_zeros(
            pred_real[-1].shape[0],
            pred_real[-1].shape[2],
            pred_real[-1].shape[3],
        )
    coeff.insert(0, highpass)
    
    return coeff


def describe_tensor(name, tensor):
    tensor = tensor.detach().float()
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.4f} max={tensor.max().item():.4f} "
        f"mean={tensor.mean().item():.4f} std={tensor.std(unbiased=False).item():.4f}"
    )


def main():
    """Main training function."""
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    
    # Create log directory
    log_dir = Path('./log')
    log_dir.mkdir(exist_ok=True)
    
    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_file = log_dir / f'{now}_train_complex.txt'
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Training parameters
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    
    # Pyramid parameters
    height = 12
    nbands = 4
    scale_factor = 2**(1/2)
    pyr_type = 1
    
    # Dataset path
    dataset_path = args.dataset_path or resolve_dataset_path()
    
    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset path not found: {dataset_path}")
        print("Set PHASENET_DATASET_PATH or place DAVIS-data under the repository root.")
        print("Using repository root as a fallback for testing...")
        dataset_path = './'
    else:
        print(f"Using dataset path: {dataset_path}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    try:
        dataset = Triplets(dataset_path, transform)
        if args.overfit_samples > 0:
            overfit_count = min(args.overfit_samples, len(dataset))
            dataset = torch.utils.data.Subset(dataset, range(overfit_count))
            print(f"Overfit/debug mode enabled: using first {overfit_count} triplets")
        print(f"Dataset loaded: {len(dataset)} triplets")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset path is correct and contains image sequences.")
        return
    
    # Initialize pyramid
    pyr = SCFpyr_PyTorch(height=height, nbands=nbands, 
                         scale_factor=scale_factor, device=device)
    
    # Define network
    model = ComplexPhaseNet(feature_dim=args.feature_dim).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = ComplexTotalLoss(
        img_weight=args.img_weight,
        residual_weight=args.residual_weight,
        residual_imag_weight=0.1,
        phase_weight=args.phase_weight,
        amp_weight=args.amp_weight,
        amp_imag_weight=args.amp_imag_loss_weight,
        phase_unit_weight=args.phase_unit_weight,
        grad_weight=args.grad_weight,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,          # FIXED: use command-line argument
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Create debug image directory
    debug_dir = args.debug_save_dir
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    total_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.overfit_samples <= 0,
            num_workers=2,
        )
        
        epoch_loss = 0.0
        num_batches = 0
        epoch_steps = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, Triplets_batch in enumerate(pbar):
            try:
                # Triplets_batch is a dict with keys 'start', 'inter', 'end'
                # Each is a list of tensors of shape [3, H, W] (RGB)
                # Stack them into batches: [batch_size, 3, H, W]
                start_batch = torch.stack([t for t in Triplets_batch['start']])   # [B,3,H,W]
                inter_batch = torch.stack([t for t in Triplets_batch['inter']])
                end_batch   = torch.stack([t for t in Triplets_batch['end']])
                
                B, C, H, W = start_batch.shape

                start_batch = start_batch.to(device)
                inter_batch = inter_batch.to(device)
                end_batch = end_batch.to(device)

                # Convert RGB to luminance Y
                start_y = (
                    0.299 * start_batch[:, 0:1, :, :] +
                    0.587 * start_batch[:, 1:2, :, :] +
                    0.114 * start_batch[:, 2:3, :, :]
                )
                inter_y = (
                    0.299 * inter_batch[:, 0:1, :, :] +
                    0.587 * inter_batch[:, 1:2, :, :] +
                    0.114 * inter_batch[:, 2:3, :, :]
                )
                end_y = (
                    0.299 * end_batch[:, 0:1, :, :] +
                    0.587 * end_batch[:, 1:2, :, :] +
                    0.114 * end_batch[:, 2:3, :, :]
                )

                # Build one pyramid triplet per sample, not per color channel
                images_list = [
                    torch.stack([start_y[i], inter_y[i], end_y[i]], dim=0)   # [3,1,H,W]
                    for i in range(B)
                ]
                batch_coeff_list = [pyr.build(img, pyr_type=pyr_type) for img in images_list]
                
                # Get complex training data (still per-channel)
                train_real, train_imag, truth_real, truth_imag, hp_start, hp_end = get_complex_input(batch_coeff_list)
                
                # Move to device and ensure float
                train_real = [t.float().to(device) for t in train_real]
                train_imag = [t.float().to(device) for t in train_imag]
                truth_real = [t.float().to(device) for t in truth_real]
                truth_imag = [t.float().to(device) for t in truth_imag]
                
                hp_start = hp_start.float().to(device)
                hp_end = hp_end.float().to(device)
                hp_mid = 0.5 * (hp_start + hp_end)
                # Forward pass
                pred_real, pred_imag = model(train_real, train_imag)
                
                # Reconstruct predicted images (still flat batch)
                pred_coeff = output_convert_complex(pred_real, pred_imag, highpass=hp_mid)
                pred_img_y = pyr.reconstruct(pred_coeff, pyr_type=pyr_type)   # usually [B, H, W]
                if pred_img_y.dim() == 3:
                    pred_img_y = pred_img_y.unsqueeze(1)   # [B,1,H,W]

                truth_img = inter_y   # [B,1,H,W]
                pred_img = pred_img_y
                
                # Compute loss (ComplexTotalLoss expects truth_img and pred_img as [B, H, W]? Actually it uses L1Loss on the full image)
                # The loss function in complex_phasenet.py takes truth_img, pred_img as 2D? Check: it uses nn.L1Loss()(truth_img, pred_img) where both are [N, H, W]? 
                # We'll pass them as they are (with channel dimension). L1Loss will work on all dimensions.
                loss = criterion(
                    truth_real, truth_imag,
                    pred_real, pred_imag,
                    truth_img, pred_img
                )
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
                total_step += 1
                epoch_steps += 1
                
                # Logging
                if total_step % args.log_interval == 0:
                    stats = criterion.last_stats
                    log_msg = (f"Step {total_step:6d} | "
                               f"Loss total={stats['total']:6.3f} "
                               f"img={stats['img']:6.3f} "
                               f"res_real={stats['residual_real']:6.3f} "
                               f"phase={stats['phase']:6.3f} "
                               f"amp={stats['amp']:6.3f}")
                    print(log_msg)
                    print(f"alpha: {model.alpha.item():.4f}, beta: {model.beta.item():.4f}")
                    print("amp_gain:", model.amp_gain.detach().cpu().numpy())
                    with open(log_file, 'a') as f:
                        f.write(log_msg + '\n')
                
                # Save debug images
                if args.save_interval > 0 and total_step % args.save_interval == 0 and debug_dir is not None:
                    with torch.no_grad():
                        # Save first sample of the batch
                        vis_truth = truth_img[0].cpu().clamp(0, 1)
                        vis_pred = pred_img[0].cpu().clamp(0, 1)
                        comparison = torch.cat([vis_truth, vis_pred], dim=2)  # concatenate horizontally
                        save_image(comparison, debug_dir / f"step_{total_step:06d}.png")
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}'
                })
                
                if args.max_steps_per_epoch > 0 and epoch_steps >= args.max_steps_per_epoch:
                    break
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f'\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}')
        
        # Update learning rate
        scheduler.step(avg_epoch_loss)
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = Path('./model') / f'{now}_complex_epoch{epoch+1}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    model_path = Path('./model') / f'{now}_complex_final.pth'
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'\nTraining completed! Final model saved: {model_path}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()