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
        "--debug-interval",
        type=int,
        default=0,
        help="If > 0, print tensor stats and optionally dump sample images every N steps.",
    )
    parser.add_argument(
        "--debug-save-dir",
        type=Path,
        default=None,
        help="Optional directory for periodic training debug images.",
    )
    parser.add_argument(
        "--phase-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to the wrapped phase loss term.",
    )
    parser.add_argument(
        "--amp-loss-weight",
        type=float,
        default=0.5,
        help="Weight applied to the amplitude supervision term.",
    )
    parser.add_argument(
        "--amp-imag-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to keeping imaginary amplitude channels near zero.",
    )
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
    
    Args:
        batch_coeff_list: List of pyramid coefficients for a batch.
                         Each element is a single pyramid built from a
                         3-frame tensor [start, inter, end], so the frame
                         axis is already packed into the first dimension of
                         every pyramid coefficient tensor.
    
    Returns:
        tuple: (train_real, train_imag, truth_real, truth_imag)
    """
    # Each item in batch_coeff_list is already a full pyramid for the
    # start/inter/end triplet, with frame index stored in dimension 0.
    # Extract real/imaginary parts level-by-level and pass them directly
    # to complex_input_convert, which expects the three frames to remain
    # stacked inside each coefficient tensor.
    train_real_batch = []
    train_imag_batch = []
    truth_real_batch = []
    truth_imag_batch = []
    
    for triplet_coeff in batch_coeff_list:
        tri_real, tri_imag = extract_complex_coefficients(triplet_coeff)
        tr_r, tr_i, truth_r, truth_i = complex_input_convert(tri_real, tri_imag)
        
        train_real_batch.append(tr_r)
        train_imag_batch.append(tr_i)
        truth_real_batch.append(truth_r)
        truth_imag_batch.append(truth_i)
    
    # Stack batch elements
    train_real = []
    train_imag = []
    truth_real = []
    truth_imag = []
    
    for level_idx in range(len(train_real_batch[0])):
        train_real.append(torch.stack([item[level_idx] for item in train_real_batch]))
        train_imag.append(torch.stack([item[level_idx] for item in train_imag_batch]))
        truth_real.append(torch.stack([item[level_idx] for item in truth_real_batch]))
        truth_imag.append(torch.stack([item[level_idx] for item in truth_imag_batch]))
    
    return train_real, train_imag, truth_real, truth_imag

def output_convert_complex(pred_real, pred_imag):
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
            phase = torch.atan2(phase_i, phase_r)
            
            # Reconstruct complex coefficient
            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            
            # Stack as complex tensor
            complex_coeff = torch.stack([real, imag], -1)
            band.append(complex_coeff)
        
        coeff.insert(0, band)
    
    # Add high-pass (zeros) on the same device/dtype as the predictions.
    coeff.insert(0, pred_real[-1].new_zeros(
        pred_real[-1].shape[0],
        pred_real[-1].shape[2],
        pred_real[-1].shape[3],
    ))
    
    return coeff


def describe_tensor(name, tensor):
    tensor = tensor.detach().float()
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.4f} max={tensor.max().item():.4f} "
        f"mean={tensor.mean().item():.4f} std={tensor.std(unbiased=False).item():.4f}"
    )


def save_debug_batch(save_dir, step, channel, batch, pred_img):
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = save_dir / f"step_{step:06d}_ch{channel}"
    save_image(batch['start'][0, channel].unsqueeze(0), prefix.with_name(f"{prefix.name}_start.png"))
    save_image(batch['inter'][0, channel].unsqueeze(0), prefix.with_name(f"{prefix.name}_truth.png"))
    save_image(batch['end'][0, channel].unsqueeze(0), prefix.with_name(f"{prefix.name}_end.png"))
    save_image(pred_img[0].detach().cpu().clamp(0, 1).unsqueeze(0), prefix.with_name(f"{prefix.name}_pred.png"))


def log_debug_stats(step, channel, truth_img, pred_img, pred_real, pred_imag):
    print(f"\n[debug] step={step} channel={channel}")
    print(describe_tensor("truth_img", truth_img))
    print(describe_tensor("pred_img", pred_img))
    if len(pred_real) > 1:
        band_amp = pred_real[1][:, :4, :, :]
        band_phase_r = pred_real[1][:, 4:8, :, :]
        band_phase_i = pred_imag[1][:, 4:8, :, :]
        phase_angle = torch.atan2(band_phase_i, band_phase_r)
        print(describe_tensor("band1_amp", band_amp))
        print(describe_tensor("band1_phase_angle", phase_angle))


def main():
    """Main training function."""
    args = parse_args()
    
    # Create log directory
    log_dir = './log/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_name = f'{now}_train_complex.txt'
    
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
    
    # Dataset path - UPDATE THIS PATH
    dataset_path = args.dataset_path or resolve_dataset_path()
    
    # Check if dataset exists
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
        v=args.phase_loss_weight,
        amp_weight=args.amp_loss_weight,
        amp_imag_weight=args.amp_imag_loss_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
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
        
        # Progress bar
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for channel in range(3):  # Process each color channel
            for n, Triplets_batch in enumerate(pbar):
                try:
                    # Get images
                    images_list = [
                        torch.stack([
                            Triplets_batch['start'][i],
                            Triplets_batch['inter'][i],
                            Triplets_batch['end'][i]
                        ]) for i in range(len(Triplets_batch['start']))
                    ]
                    
                    # Build pyramids
                    batch_coeff_list = [
                        pyr.build(image[:, channel, :, :].unsqueeze(1).to(device), 
                                 pyr_type=pyr_type)
                        for image in images_list
                    ]
                    
                    # Get complex training data
                    train_real, train_imag, truth_real, truth_imag = \
                        get_complex_input(batch_coeff_list)
                    
                    # Move to device
                    train_real = [t.float().to(device) for t in train_real]
                    train_imag = [t.float().to(device) for t in train_imag]
                    truth_real = [t.float().to(device) for t in truth_real]
                    truth_imag = [t.float().to(device) for t in truth_imag]
                    
                    # Forward pass
                    pred_real, pred_imag = model(train_real, train_imag)
                    
                    # Reconstruct image
                    truth_img = Triplets_batch['inter'][:, channel, :, :].to(device)
                    pred_coeff = output_convert_complex(pred_real, pred_imag)
                    pred_img = pyr.reconstruct(pred_coeff, pyr_type=pyr_type)
                    
                    # Compute loss
                    loss = criterion(
                        truth_real,
                        truth_imag,
                        pred_real,
                        pred_imag,
                        truth_img,
                        pred_img,
                    )
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    num_batches += 1
                    total_step += 1
                    
                    if args.debug_interval > 0 and total_step % args.debug_interval == 0:
                        log_debug_stats(
                            total_step,
                            channel,
                            truth_img,
                            pred_img,
                            pred_real,
                            pred_imag,
                        )
                        if args.debug_save_dir is not None:
                            save_debug_batch(
                                args.debug_save_dir,
                                total_step,
                                channel,
                                Triplets_batch,
                                pred_img,
                            )

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}',
                        'channel': channel
                    })
                    
                    # Log periodically
                    if total_step % 10 == 0:
                        log_msg = (f'Epoch [{epoch+1}/{num_epochs}], '
                                  f'Channel [{channel}], '
                                  f'Step [{total_step}], '
                                  f'Loss: {loss.item():.4f}\n')
                        
                        with open(os.path.join(log_dir, log_name), 'a') as f:
                            f.write(log_msg)
                    if args.max_steps_per_epoch > 0 and epoch_steps >= args.max_steps_per_epoch:
                        break
                except Exception as e:
                    print(f"\nError in batch {n}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            if args.max_steps_per_epoch > 0 and epoch_steps >= args.max_steps_per_epoch:
                break
        # Epoch statistics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f'\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}')
        
        # Update learning rate
        scheduler.step(avg_epoch_loss)
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'./model/{now}_complex_epoch{epoch+1}.pth'
            os.makedirs('./model', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    model_path = f'./model/{now}_complex_final.pth'
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
