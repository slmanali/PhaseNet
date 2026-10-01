import argparse
import os
from pathlib import Path
import time

import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import Triplets
from utils.reproducibility import (configure_reproducibility, create_run_directory,
    dataset_sample_ids, environment_metadata, load_checkpoint, make_data_generator,
    ordered_hash, save_training_checkpoint, seed_worker, write_metadata)
from utils.davis import (davis_image_root, load_davis_train_val,
                         print_davis_split, resolve_davis_root)
from net.complex_phasenet_safe_baseline import (
    ComplexPhaseNetSafe,
    ComplexTotalLoss,
    complex_input_convert,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train safer complex-valued PhaseNet baseline.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--davis-root", type=str, default=None)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--debug-save-dir", type=Path, default=Path("./debug_complex_safe"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--deterministic", choices=("strict", "warn", "off"), default="strict")
    parser.add_argument("--run-root", type=Path, default=Path("./runs_seeded"))
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume a full checkpoint at a completed epoch boundary.")

    parser.add_argument("--img-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--residual-imag-weight", type=float, default=0.05)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--amp-weight", type=float, default=0.0)
    parser.add_argument("--amp-imag-loss-weight", type=float, default=0.0)
    parser.add_argument("--phase-unit-weight", type=float, default=0.01)
    parser.add_argument("--grad-weight", type=float, default=0.1)
    parser.add_argument("--phase-correction-scale", type=float, default=0.1)
    parser.add_argument("--residual-correction-scale", type=float, default=0.1)
    parser.add_argument("--convolution", choices=("complex", "separate_real", "unrestricted_real"), default="complex")
    return parser.parse_args()


def resolve_dataset_path():
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


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_complex_coefficients(pyr_coeff):
    coeff_real = []
    coeff_imag = []
    for level in pyr_coeff:
        if isinstance(level, list):
            level_real = []
            level_imag = []
            for band in level:
                if band.dim() == 4 and band.shape[-1] == 2:
                    level_real.append(band[..., 0])
                    level_imag.append(band[..., 1])
                else:
                    level_real.append(band)
                    level_imag.append(torch.zeros_like(band))
            coeff_real.append(level_real)
            coeff_imag.append(level_imag)
        else:
            coeff_real.append(level)
            coeff_imag.append(torch.zeros_like(level))
    return coeff_real, coeff_imag


def get_complex_input(batch_coeff_list):
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

    train_real, train_imag, truth_real, truth_imag = [], [], [], []
    for level_idx in range(len(train_real_batch[0])):
        train_real.append(torch.stack([item[level_idx] for item in train_real_batch]))
        train_imag.append(torch.stack([item[level_idx] for item in train_imag_batch]))
        truth_real.append(torch.stack([item[level_idx] for item in truth_real_batch]))
        truth_imag.append(torch.stack([item[level_idx] for item in truth_imag_batch]))

    highpass_start = torch.stack(highpass_start_batch)
    highpass_end = torch.stack(highpass_end_batch)
    return train_real, train_imag, truth_real, truth_imag, highpass_start, highpass_end


def output_convert_complex(pred_real, pred_imag, highpass=None):
    coeff = []
    coeff.append(pred_real[0].squeeze(1))

    for i in range(1, len(pred_real)):
        bands_num = pred_real[i].shape[1] // 2
        band = []
        for j in range(bands_num):
            amp_r = pred_real[i][:, j, :, :]
            amp_i = pred_imag[i][:, j, :, :]
            phase_r = pred_real[i][:, j + bands_num, :, :]
            phase_i = pred_imag[i][:, j + bands_num, :, :]

            amp = torch.sqrt(amp_r ** 2 + amp_i ** 2 + 1e-8)
            phase = torch.atan2(phase_i + 1e-8, phase_r + 1e-8)
            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            band.append(torch.stack([real, imag], -1))
        coeff.insert(0, band)

    if highpass is None:
        highpass = pred_real[-1].new_zeros(
            pred_real[-1].shape[0], pred_real[-1].shape[2], pred_real[-1].shape[3]
        )
    coeff.insert(0, highpass)
    return coeff


def main():
    args = parse_args()
    configure_reproducibility(args.seed, args.deterministic)
    torch.autograd.set_detect_anomaly(True)

    log_dir = Path('./log')
    log_dir.mkdir(exist_ok=True)
    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_file = log_dir / f'{now}_train_complex_safe.txt'

    device = resolve_device(args.device)
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    dataset_path = args.dataset_path or (None if args.davis_root else resolve_dataset_path())
    davis_root = resolve_davis_root(args.davis_root, dataset_path)
    dataset_path = str(davis_image_root(davis_root, args.dataset_path))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    print(f'Using dataset path: {dataset_path}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_sequences, val_sequences = load_davis_train_val(davis_root)
    assert set(train_sequences).isdisjoint(set(val_sequences))
    dataset = Triplets(dataset_path, transform, allowed_sequences=train_sequences)
    print_davis_split(args.split, train_sequences, len(dataset))
    if args.overfit_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.overfit_samples, len(dataset))))
        print(f'Overfit/debug mode enabled: using first {len(dataset)} triplets')
    print(f'Dataset loaded: {len(dataset)} triplets')
    config_name = args.config_name or "complex-safe-{args.feature_dim}"
    run_dir = create_run_directory(args.run_root, config_name, args.seed, resume=args.resume is not None)
    data_generator = make_data_generator(args.seed)
    split_hash = ordered_hash(dataset_sample_ids(dataset))


    pyr = SCFpyr_PyTorch(height=12, nbands=4, scale_factor=2 ** (1 / 2), device=device)
    pyr_type = 1

    reference_model = None
    if args.convolution != "complex":
        reference_model = ComplexPhaseNetSafe(
            feature_dim=args.feature_dim,
            phase_correction_scale=args.phase_correction_scale,
            residual_correction_scale=args.residual_correction_scale,
            convolution="complex",
        )
    model = ComplexPhaseNetSafe(
        feature_dim=args.feature_dim,
        phase_correction_scale=args.phase_correction_scale,
        residual_correction_scale=args.residual_correction_scale,
        convolution=args.convolution,
    )
    if reference_model is not None:
        model.initialize_ablation_from(reference_model)
        del reference_model
    model = model.to(device)
    print('\nModel architecture:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    criterion = ComplexTotalLoss(
        img_weight=args.img_weight,
        residual_weight=args.residual_weight,
        residual_imag_weight=args.residual_imag_weight,
        phase_weight=args.phase_weight,
        amp_weight=args.amp_weight,
        amp_imag_weight=args.amp_imag_loss_weight,
        phase_unit_weight=args.phase_unit_weight,
        grad_weight=args.grad_weight,
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

    start_epoch = 0
    total_step = 0
    if args.resume is not None:
        resumed = load_checkpoint(args.resume, model, optimizer, scheduler, data_generator, map_location=device)
        if resumed.get("legacy"):
            raise ValueError("training resume requires a full checkpoint, not a legacy state_dict")
        start_epoch, total_step = resumed["epoch"], resumed["global_step"]
    metadata = {
        "status": "running", "seed": args.seed, "configuration": config_name,
        "arguments": vars(args), "split_sample_sha256": split_hash,
        "sample_count": len(dataset), "environment": environment_metadata(),
        "checkpoint_selection": "final_completed_epoch", "completed_epochs": start_epoch,
        "completed_steps": total_step, "resume_boundary": "completed_epoch",
        "optimizer": {"name": "Adam", "lr": args.learning_rate,
                      "betas": [0.9, 0.999], "weight_decay": 1e-5},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5,
                      "patience": 2, "mode": "min"},
        "input_size": [256, 256], "precision": "fp32",
        "convolution_replacement_scope": "feature and prediction-head convolutions",
        "initialization": ("native_complex" if args.convolution == "complex" else
                           "matching_A_B" if args.convolution == "separate_real" else
                           "complex_block_matrix"),
    }
    write_metadata(run_dir / "metadata.json", metadata)

    if args.debug_save_dir is not None:
        args.debug_save_dir.mkdir(parents=True, exist_ok=True)

    print('\nStarting training...')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.overfit_samples <= 0,
            num_workers=args.num_workers,
            generator=data_generator,
            worker_init_fn=seed_worker,
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
                vis_pred_channels = []

                for channel in range(3):
                    start_ch = start_batch[:, channel:channel + 1, :, :]
                    inter_ch = inter_batch[:, channel:channel + 1, :, :]
                    end_ch = end_batch[:, channel:channel + 1, :, :]

                    images_list = [torch.stack([start_ch[i], inter_ch[i], end_ch[i]]).to(device) for i in range(B)]
                    batch_coeff_list = [pyr.build(img, pyr_type=pyr_type) for img in images_list]
                    train_real, train_imag, truth_real, truth_imag, hp_start, hp_end = get_complex_input(batch_coeff_list)

                    train_real = [t.float().to(device) for t in train_real]
                    train_imag = [t.float().to(device) for t in train_imag]
                    truth_real = [t.float().to(device) for t in truth_real]
                    truth_imag = [t.float().to(device) for t in truth_imag]
                    hp_mid = 0.5 * (hp_start.float().to(device) + hp_end.float().to(device))

                    pred_real, pred_imag = model(train_real, train_imag)
                    pred_coeff = output_convert_complex(pred_real, pred_imag, highpass=hp_mid)
                    pred_img_ch = pyr.reconstruct(pred_coeff, pyr_type=pyr_type)
                    if pred_img_ch.dim() == 3:
                        pred_img_ch = pred_img_ch.unsqueeze(1)

                    loss_ch = criterion(
                        truth_real, truth_imag,
                        pred_real, pred_imag,
                        inter_ch, pred_img_ch,
                    )
                    channel_losses.append(loss_ch)
                    vis_pred_channels.append(pred_img_ch.detach().cpu().clamp(0, 1))

                loss = sum(channel_losses) / len(channel_losses)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch + 1}, batch {batch_idx}: {loss}")

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"nonfinite gradient norm at epoch {epoch + 1}, batch {batch_idx}")
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
                        f"res_real={stats['residual_real']:6.3f} res_imag={stats['residual_imag']:6.3f} "
                        f"phase={stats['phase']:6.3f} phase_unit={stats['phase_unit']:6.3f} amp={stats['amp']:6.3f}"
                    )
                    print(log_msg)
                    print(f"alpha: {model.alpha.item():.4f}, beta: {model.beta.item():.4f}")
                    with open(log_file, 'a') as f:
                        f.write(log_msg + '\n')

                if args.save_interval > 0 and total_step % args.save_interval == 0:
                    with torch.no_grad():
                        vis_truth = inter_batch[0].cpu().clamp(0, 1)
                        vis_pred = torch.cat([
                            vis_pred_channels[0][0],
                            vis_pred_channels[1][0],
                            vis_pred_channels[2][0],
                        ], dim=0)
                        comparison = torch.cat([vis_truth, vis_pred], dim=2)
                        save_image(comparison, args.debug_save_dir / f"step_{total_step:06d}.png")

                pbar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{epoch_loss / num_batches:.4f}')

                if args.max_steps_per_epoch > 0 and epoch_steps >= args.max_steps_per_epoch:
                    break

            except Exception as e:
                raise RuntimeError(f"training failed at epoch {epoch + 1}, batch {batch_idx}") from e

        if num_batches == 0:
            raise RuntimeError(f"epoch {epoch + 1} completed with zero successful batches")
        avg_epoch_loss = epoch_loss / num_batches
        print(f'\nEpoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}')
        scheduler.step(avg_epoch_loss)
        checkpoint_path = run_dir / "checkpoint_last.pth"
        save_training_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch + 1,
                                 total_step, vars(args), data_generator)
        metadata.update(status="running", completed_epochs=epoch + 1,
                        completed_steps=total_step, checkpoint=str(checkpoint_path))
        write_metadata(run_dir / "metadata.json", metadata)

        if (epoch + 1) % 2 == 0:
            checkpoint_path = run_dir / f'legacy_epoch{epoch + 1}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    model_path = run_dir / 'model_final_state_dict.pth'
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    metadata.update(status="completed", checkpoint=str(run_dir / "checkpoint_last.pth"))
    write_metadata(run_dir / "metadata.json", metadata)
    print(f'\nTraining completed! Final model saved: {model_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception:
        # Research runs must terminate nonzero; metadata/checkpoint remain inspectable.
        raise
