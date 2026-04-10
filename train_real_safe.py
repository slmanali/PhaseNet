"""
Safer training script for the real-valued PhaseNet baseline.
Keeps the original real-model logic, but adds:
- feature_dim argument
- configurable learning rate
- gradient loss and weighted components
- detailed logging
- same debug/save cadence style as complex training
"""

import argparse
import os
from pathlib import Path
import time

import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, get_input, Total_loss, output_convert


def parse_args():
    parser = argparse.ArgumentParser(description="Train the safer real-valued PhaseNet baseline.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--debug-save-dir", type=Path, default=Path("./debug_real_safe"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--img-weight", type=float, default=1.0)
    parser.add_argument("--phase-weight", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.2)
    parser.add_argument("--amp-weight", type=float, default=0.0)
    return parser.parse_args()


def resolve_dataset_path():
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
    return str(candidate_paths[0] if candidate_paths else repo_root)


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)

    log_dir = Path('./log')
    log_dir.mkdir(exist_ok=True)
    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_file = log_dir / f'{now}_train_real_safe.txt'

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    dataset_path = args.dataset_path or resolve_dataset_path()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    print(f"Using dataset path: {dataset_path}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = Triplets(dataset_path, transform)
    if args.overfit_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.overfit_samples, len(dataset))))
        print(f"Overfit/debug mode enabled: using first {len(dataset)} triplets")
    print(f"Dataset loaded: {len(dataset)} triplets")

    pyr = SCFpyr_PyTorch(height=12, nbands=4, scale_factor=2 ** (1 / 2), device=device)
    pyr_type = 1

    model = PhaseNet(feature_dim=args.feature_dim).to(device)
    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = Total_loss(
        img_weight=args.img_weight,
        phase_weight=args.phase_weight,
        residual_weight=args.residual_weight,
        grad_weight=args.grad_weight,
        amp_weight=args.amp_weight,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    if args.debug_save_dir is not None:
        args.debug_save_dir.mkdir(parents=True, exist_ok=True)

    total_step = 0
    print("\nStarting training...")

    for epoch in range(args.epochs):
        model.train()
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.overfit_samples <= 0,
            num_workers=args.num_workers,
        )

        epoch_loss = 0.0
        num_batches = 0
        epoch_steps = 0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch_idx, triplets_batch in enumerate(pbar):
            try:
                start_batch = torch.stack([t for t in triplets_batch['start']]).to(device)
                inter_batch = torch.stack([t for t in triplets_batch['inter']]).to(device)
                end_batch = torch.stack([t for t in triplets_batch['end']]).to(device)
                B = start_batch.shape[0]

                channel_losses = []
                vis_pred_ch = None

                for channel in range(3):
                    start_ch = start_batch[:, channel:channel + 1, :, :]
                    inter_ch = inter_batch[:, channel:channel + 1, :, :]
                    end_ch = end_batch[:, channel:channel + 1, :, :]

                    images_list = [torch.stack([start_ch[i], inter_ch[i], end_ch[i]]).to(device) for i in range(B)]
                    batch_coeff_list = [pyr.build(img, pyr_type=pyr_type) for img in images_list]
                    train_coeff, truth_coeff, amp_scales = get_input(batch_coeff_list)
                    train_coeff = [t.float().to(device) for t in train_coeff]
                    truth_coeff = [t.float().to(device) for t in truth_coeff]

                    pred_coeff = model(train_coeff)
                    pred_img_ch = pyr.reconstruct(output_convert(pred_coeff, amp_scales=amp_scales), pyr_type=pyr_type)
                    if pred_img_ch.dim() == 3:
                        pred_img_ch = pred_img_ch.unsqueeze(1)

                    loss_ch = criterion(truth_coeff, pred_coeff, inter_ch, pred_img_ch)
                    channel_losses.append(loss_ch)

                    if vis_pred_ch is None:
                        vis_pred_ch = pred_img_ch.detach()

                loss = sum(channel_losses) / len(channel_losses)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                total_step += 1
                epoch_steps += 1

                if total_step % args.log_interval == 0:
                    stats = criterion.last_stats
                    log_msg = (
                        f"Step {total_step:6d} | Loss total={stats['total']:6.3f} "
                        f"img={stats['img']:6.3f} grad={stats['grad']:6.3f} "
                        f"residual={stats['residual']:6.3f} phase={stats['phase']:6.3f} "
                        f"amp={stats['amp']:6.3f}"
                    )
                    print(log_msg)
                    print(f"alpha: {model.alpha.item():.4f}, beta: {model.beta.item():.4f}")
                    with open(log_file, 'a') as f:
                        f.write(log_msg + '\n')

                if args.save_interval > 0 and total_step % args.save_interval == 0 and vis_pred_ch is not None:
                    with torch.no_grad():
                        vis_truth = inter_batch[0].cpu().clamp(0, 1)
                        vis_pred = torch.cat([
                            vis_pred_ch[0, 0].cpu().clamp(0, 1).unsqueeze(0),
                            vis_pred_ch[0, 0].cpu().clamp(0, 1).unsqueeze(0),
                            vis_pred_ch[0, 0].cpu().clamp(0, 1).unsqueeze(0),
                        ], dim=0)
                        comparison = torch.cat([vis_truth, vis_pred], dim=2)
                        save_image(comparison, args.debug_save_dir / f"step_{total_step:06d}.png")

                pbar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{epoch_loss / num_batches:.4f}')

                if args.max_steps_per_epoch > 0 and epoch_steps >= args.max_steps_per_epoch:
                    break

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f'\nEpoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}')
        scheduler.step(avg_epoch_loss)

        if (epoch + 1) % 2 == 0:
            checkpoint_path = Path('./model') / f'{now}_real_safe_epoch{epoch + 1}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    model_path = Path('./model') / f'{now}_real_safe_final.pth'
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'\nTraining completed! Final model saved: {model_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
