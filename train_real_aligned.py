"""
Training script for Real-Valued PhaseNet (refactored to match train_complex.py structure).

Loads video triplets, processes through steerable pyramids, and trains PhaseNet
to predict interpolated frames.
"""

import argparse
import os
from pathlib import Path
import time

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, get_input, Total_loss, output_convert



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the real-valued PhaseNet model (aligned with train_complex.py)."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Adam learning rate.")
    parser.add_argument("--feature-dim", type=int, default=64, help="Model feature dimension.")
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
        help="If > 0, train on only the first N triplets for debugging.",
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
    parser.add_argument("--img-weight", type=float, default=8.0)
    parser.add_argument("--residual-weight", type=float, default=0.2)
    parser.add_argument("--phase-weight", type=float, default=0.15)
    parser.add_argument("--amp-weight", type=float, default=0.4)
    parser.add_argument("--grad-weight", type=float, default=2.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


def resolve_dataset_path():
    """Resolve DAVIS dataset path from env/configured/common locations."""
    candidate_paths = []

    env_path = os.environ.get('PHASENET_DATASET_PATH')
    if env_path:
        candidate_paths.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parent
    candidate_paths.extend([
        Path('/home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p'),
        Path('/home/lj/Documents/code/python/DAVIS/JPEGImages/480p'),
        repo_root / 'DAVIS-data' / 'DAVIS' / 'JPEGImages' / '480p',
    ])

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    print(f"WARNING: No dataset found in standard locations.")
    print(f"Set PHASENET_DATASET_PATH or use --dataset-path")
    return str(candidate_paths[0] if candidate_paths else repo_root)


def resolve_device(device_arg):
    """Resolve PyTorch device."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Main training function."""
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)

    # Setup logging
    log_dir = Path('./log')
    log_dir.mkdir(exist_ok=True)

    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_file = log_dir / f'{now}_train.txt'

    # Device setup
    device = resolve_device(args.device)
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
    scale_factor = 2 ** (1 / 2)
    pyr_type = 1

    # Dataset path
    dataset_path = args.dataset_path or resolve_dataset_path()

    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset path not found: {dataset_path}")
        print("Set PHASENET_DATASET_PATH or place DAVIS-data under the repository root.")
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
            print(f"Overfit/debug mode: using first {overfit_count} triplets")
        print(f"Dataset loaded: {len(dataset)} triplets")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please verify the dataset path contains image sequences.")
        return

    # Initialize pyramid
    pyr = SCFpyr_PyTorch(
        height=height,
        nbands=nbands,
        scale_factor=scale_factor,
        device=device
    )

    # Define network
    model = PhaseNet(feature_dim=args.feature_dim).to(device)
    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = Total_loss(
        img_weight=args.img_weight,
        residual_weight=args.residual_weight,
        phase_weight=args.phase_weight,
        amp_weight=args.amp_weight,
        grad_weight=args.grad_weight,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Create debug image directory
    if args.debug_save_dir is not None:
        args.debug_save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("Starting training...\n")
    total_step = 0

    for epoch in range(num_epochs):
        model.train()
        
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.overfit_samples <= 0,
            num_workers=args.num_workers,
        )

        epoch_loss = 0.0
        num_batches = 0
        epoch_steps = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')


        for batch_idx, triplets_batch in enumerate(pbar):
            try:
                start_batch = torch.stack([t for t in triplets_batch['start']]).to(device)
                inter_batch = torch.stack([t for t in triplets_batch['inter']]).to(device)
                end_batch = torch.stack([t for t in triplets_batch['end']]).to(device)

                B, C, H, W = start_batch.shape

                channel_losses = []
                channel_stats = []
                debug_pred = None

                for channel in range(3):
                    start_ch = start_batch[:, channel:channel+1, :, :]
                    inter_ch = inter_batch[:, channel:channel+1, :, :]
                    end_ch = end_batch[:, channel:channel+1, :, :]

                    images_list = [
                        torch.stack([start_ch[i], inter_ch[i], end_ch[i]]).to(device)
                        for i in range(B)
                    ]
                    batch_coeff_list = [
                        pyr.build(img, pyr_type=pyr_type)
                        for img in images_list
                    ]

                    train_coeff, truth_coeff, amp_scales = get_input(batch_coeff_list)
                    train_coeff = [t.float().to(device) for t in train_coeff]
                    truth_coeff = [t.float().to(device) for t in truth_coeff]

                    pred_coeff = model(train_coeff)

                    pred_img_ch = pyr.reconstruct(
                        output_convert(pred_coeff, amp_scales=amp_scales),
                        pyr_type=pyr_type
                    )
                    if pred_img_ch.dim() == 3:
                        pred_img_ch = pred_img_ch.unsqueeze(1)

                    truth_img_ch = inter_ch
                    loss_ch = criterion(
                        truth_coeff,
                        pred_coeff,
                        truth_img_ch,
                        pred_img_ch
                    )
                    channel_losses.append(loss_ch)
                    channel_stats.append(dict(criterion.last_stats))

                    if channel == 0:
                        debug_pred = pred_img_ch.detach()

                loss = sum(channel_losses) / len(channel_losses)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                avg_stats = {}
                for key in channel_stats[0].keys():
                    avg_stats[key] = sum(stat[key] for stat in channel_stats) / len(channel_stats)

                epoch_loss += loss.item()
                num_batches += 1
                total_step += 1
                epoch_steps += 1

                if total_step % args.log_interval == 0:
                    log_msg = (
                        f"Step {total_step:6d} | "
                        f"Loss total={avg_stats['total']:6.3f} "
                        f"img={avg_stats['img']:6.3f} "
                        f"grad={avg_stats['grad']:6.3f} "
                        f"residual={avg_stats['residual']:6.3f} "
                        f"phase={avg_stats['phase']:6.3f} "
                        f"amp={avg_stats['amp']:6.3f}"
                    )
                    print(log_msg)
                    print(f"alpha: {model.alpha.item():.4f}, beta: {model.beta.item():.4f}")

                    with open(log_file, 'a') as f:
                        f.write(log_msg + '\n')

                if args.save_interval > 0 and total_step % args.save_interval == 0:
                    with torch.no_grad():
                        vis_truth = inter_batch[0, 0:1, :, :].cpu().clamp(0, 1)
                        vis_pred = debug_pred[0].cpu().clamp(0, 1)
                        comparison = torch.cat([vis_truth, vis_pred], dim=2)
                        save_image(
                            comparison,
                            args.debug_save_dir / f"step_{total_step:06d}.png"
                        )

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
        print(f'\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}\n')

        # Update learning rate
        scheduler.step(avg_epoch_loss)

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = Path('./model') / f'{now}_epoch{epoch+1}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    # Save final model
    model_path = Path('./model') / f'{now}_final.pth'
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
