"""
Evaluation script for PhaseNet checkpoints.

Loads a final ``state_dict`` saved by ``train.py`` and evaluates it on
triplets generated from the DAVIS-style dataset layout.
"""

import argparse
import math
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
from net.phasenet import PhaseNet, Triplets, get_input, output_convert
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from test_complex_safe_baseline import UCF101Triplets, MiddleburyTriplets
from utils.davis import (davis_image_root, load_davis_train_val,
                         print_davis_split, resolve_davis_root)
from utils.metrics import (compute_l1 as shared_l1, compute_mse as shared_mse,
                           compute_psnr as shared_psnr, compute_ssim as shared_ssim,
                           compute_lpips as shared_lpips, compute_real_pce)
from utils.best_metrics import BestMetricsTracker, merge_best_summaries
from utils.snufilm import (SNUFILMTriplets, SNU_MODES, snufilm_modes,
                           write_snufilm_results)

# ============================================================================
# REUSABLE METRIC FUNCTIONS (Same as test_complex.py)
# ============================================================================

def compute_psnr(pred, target):
    """Compute PSNR between pred and target."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """
    Compute mean SSIM between two batches of images (pure PyTorch).
    """
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32, device=pred.device)
        coords = coords - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)

    kernel = gaussian_window(window_size, sigma).repeat(pred.shape[1], 1, 1, 1)
    
    mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=pred.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim_map.mean(dim=[1, 2, 3]).mean().item()


def compute_lpips(pred, target, net_type='alex', device='cuda'):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity)."""
    loss_fn = lpips.LPIPS(net=net_type, verbose=False).to(device)
    
    # lpips expects [-1, 1] range
    pred_normalized = 2 * pred - 1
    target_normalized = 2 * target - 1
    
    with torch.no_grad():
        distance = loss_fn(pred_normalized, target_normalized)
    
    return distance.mean().item()

def denormalize_coefficients(coeff_list, amp_scales_list):
    """
    Denormalize coefficients using per-level amplitude scales.
    
    Args:
        coeff_list: list of tensors from model (normalized)
                   - coeff_list[0]: residual [B, 1, H, W]
                   - coeff_list[i>0]: band [B, 2*n_orient, H, W]
        amp_scales_list: list of scales from get_input()
                        - amp_scales_list[i] corresponds to coeff_list[i+1]
    
    Returns:
        list of denormalized tensors
    """
    denorm_coeff = []
    
    for i, coeff in enumerate(coeff_list):
        if i == 0:  # residual, no denormalization
            denorm_coeff.append(coeff)
        else:
            n_orient = coeff.shape[1] // 2
            amps = coeff[:, :n_orient, :, :]
            phases = coeff[:, n_orient:, :, :]
            
            # amp_scales_list[i-1] is the scale for coeff_list[i]
            scale_idx = i - 1
            if scale_idx < len(amp_scales_list):
                scale = amp_scales_list[scale_idx]  # [B, n_orient, H_scale, W_scale]
                H, W = amps.shape[2], amps.shape[3]
                
                # Resize scale to match amplitude spatial dimensions if needed
                if scale.shape[2] != H or scale.shape[3] != W:
                    scale = F.interpolate(
                        scale,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                
                amps_denorm = amps * scale
            else:
                amps_denorm = amps
            
            denorm_coeff.append(torch.cat([amps_denorm, phases], dim=1))
    
    return denorm_coeff


def convert_coeff_to_complex_format(coeff_list):
    """
    Convert denormalized coefficient list to real/imag format for PCE.
    
    Args:
        coeff_list: list of tensors [B, C, H, W]
                   - level 0: [B, 1, H, W] (residual)
                   - levels i>0: [B, 2*n_orient, H, W] (amps + phases)
    
    Returns:
        (real_parts, imag_parts): lists for compute_pce()
    """
    real_parts = []
    imag_parts = []
    
    for lvl in coeff_list:
        if lvl.dim() == 3:
            lvl = lvl.unsqueeze(1)
        
        if lvl.shape[1] == 1:  # residual
            real_parts.append(lvl)
            imag_parts.append(torch.zeros_like(lvl))
        else:  # band level
            n_orient = lvl.shape[1] // 2
            amps = lvl[:, :n_orient, :, :]
            phases = lvl[:, n_orient:, :, :]
            
            real_parts.append(amps)
            imag_parts.append(torch.zeros_like(amps))
            real_parts.append(torch.cos(phases))
            imag_parts.append(torch.sin(phases))
    
    return real_parts, imag_parts


def compute_pce(truth_real, truth_imag, pred_real, pred_imag, amp_eps=1e-4):
    """
    Phase Coherence Error (PCE):
        mean_pixel( wrapped_abs(phase_pred - phase_gt) )

    Uses only band levels (skip residual / low-pass level 0).
    Returns mean angular error in radians, in [0, pi].

    amp_eps:
        optional threshold to ignore pixels where GT amplitude is too small,
        because phase there is unstable / not meaningful.
    """
    total_error = 0.0
    total_count = 0

    for i in range(1, len(truth_real)):
        n_orient = truth_real[i].shape[1] // 2

        # Ground-truth phase channels
        truth_phase_r = truth_real[i][:, n_orient:, :, :]
        truth_phase_i = truth_imag[i][:, n_orient:, :, :]

        # Predicted phase channels
        pred_phase_r = pred_real[i][:, n_orient:, :, :]
        pred_phase_i = pred_imag[i][:, n_orient:, :, :]

        # Angles in [-pi, pi]
        truth_angle = torch.atan2(truth_phase_i, truth_phase_r)
        pred_angle = torch.atan2(pred_phase_i, pred_phase_r)

        # Wrapped angular difference in [-pi, pi]
        phase_diff = torch.atan2(
            torch.sin(pred_angle - truth_angle),
            torch.cos(pred_angle - truth_angle)
        ).abs()   # [B, n_orient, H, W], now in [0, pi]

        # Optional masking by GT amplitude magnitude
        truth_amp = truth_real[i][:, :n_orient, :, :]
        valid_mask = truth_amp > amp_eps

        total_error += phase_diff[valid_mask].sum().item()
        total_count += valid_mask.sum().item()

    if total_count == 0:
        return 0.0

    return total_error / total_count

def normalize_for_visualization(image):
    """
    Convert a single image tensor to [0, 1] range for visualization.
    Preserves raw tensor for metrics while stretching saved visualization.
    """
    image = image.detach().float()
    image_min = image.min()
    image_max = image.max()

    if not torch.isfinite(image_min) or not torch.isfinite(image_max):
        return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

    dynamic_range = image_max - image_min
    if dynamic_range <= 1e-8:
        return torch.zeros_like(image)

    return (image - image_min) / dynamic_range


# ============================================================================
# BEST METRICS TRACKER
# ============================================================================

# ============================================================================
# ARGUMENT PARSING & DEVICE SETUP
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PhaseNet model on triplet data."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a final *.pth checkpoint file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the DAVIS-style dataset root.",
    )
    parser.add_argument("--davis-root", type=str, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dataset-type", choices=("auto", "davis", "ucf101", "middlebury", "snufilm"), default="auto")
    parser.add_argument("--snu-mode", choices=(*SNU_MODES, "all"), default="easy")
    parser.add_argument("--image-size", choices=("native", "256"), default="256",
                        help="SNU-FILM input resolution; native never resizes or tiles.")
    parser.add_argument("--model-name", default=None,
                        help="Paper CSV label (defaults to PhaseNet-default/big from feature_dim).")
    parser.add_argument("--metrics-dir", type=Path, default=Path("."),
                        help="Directory for metrics_snufilm.csv/json.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of triplets to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory where best metric images are saved.",
    )
    parser.add_argument("--best-k", type=int, default=1,
                        help="Number of best SNU-FILM samples retained per metric.")
    parser.add_argument("--sample-indices",
                        help="Comma-separated zero-based official SNU-FILM indices to evaluate.")
    parser.add_argument("--save-all", action="store_true",
                        help="Save every input/target/prediction (for qualitative analysis).")
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="Feature dimension used when training the model.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device, feature_dim):
    """Load model checkpoint (state_dict only)."""
    checkpoint = torch.load(model_path, map_location=device)
    model = PhaseNet(feature_dim=feature_dim).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch")
    else:
        model.load_state_dict(checkpoint)
        epoch = None

    model.eval()
    return model, epoch


# ============================================================================
# EVALUATION LOOP
# ============================================================================

def evaluate(model, dataloader, device, save_dir=None, max_samples=None, save_all=False,
             best_k=1, model_name="model", mode="mode"):

    """Evaluate model on dataloader and optionally save best metric images."""
    pyr = SCFpyr_PyTorch(
        height=12,
        nbands=4,
        scale_factor=2 ** (1 / 2),
        device=device,
    )
    pyr_type = 1

    l1_total = 0.0
    mse_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    pce_total = 0.0
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device) 
    processed = 0

    tracker = BestMetricsTracker(save_dir, model_name, mode, best_k) if save_dir else None
    progress = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch in progress:
            if max_samples is not None and processed >= max_samples:
                break

            batch_size = len(batch["start"])
            images_list = [
                torch.stack([batch["start"][i], batch["inter"][i], batch["end"][i]])
                for i in range(batch_size)
            ]

            recon_channels = []
            pce_batch_sum = 0.0
            
            for channel in range(3):
                batch_coeff_list = [
                    pyr.build(
                        image[:, channel, :, :].unsqueeze(1).to(device),
                        pyr_type=pyr_type,
                    )
                    for image in images_list
                ]

                # Prepare inputs for model
                train_coeff, truth_coeff, amp_scales = get_input(batch_coeff_list)

                # Forward pass
                pred_coeff = model(train_coeff)
                # print(f"DEBUG: amp_scales length = {len(amp_scales)}")
                # for i, scale in enumerate(amp_scales):
                #     print(f"  Level {i}: scale shape = {scale.shape}")
                # print(f"DEBUG: pred_coeff length = {len(pred_coeff)}")
                # for i, coeff in enumerate(pred_coeff):
                #     print(f"  Level {i}: coeff shape = {coeff.shape}") 
                
                # ✓ FIX: Denormalize using amp_scales (stays as list[Tensor])
                # amp_scales_dict = {i: amp_scales[i] for i in range(len(amp_scales))}
                pce_channel = compute_real_pce(truth_coeff, pred_coeff)
                pce_batch_sum += pce_channel

                # Reconstruction: convert back to output_convert format for pyr.reconstruct()
                pred_coeff_for_recon = output_convert(pred_coeff, amp_scales=amp_scales)
                pred_img = pyr.reconstruct(pred_coeff_for_recon, pyr_type=pyr_type)
                recon_channels.append(pred_img.unsqueeze(1))

            raw_pred_batch = torch.cat(recon_channels, dim=1)
            pred_batch = raw_pred_batch.clamp(0, 1)
            truth_batch = batch["inter"].to(device)

            if max_samples is not None:
                remaining = max_samples - processed
                pred_batch = pred_batch[:remaining]
                truth_batch = truth_batch[:remaining]

            # Compute metrics
            l1_batch = shared_l1(pred_batch, truth_batch)
            mse_batch = shared_mse(pred_batch, truth_batch)
            psnr_batch = shared_psnr(pred_batch, truth_batch)
            ssim_batch = shared_ssim(pred_batch, truth_batch)
            lpips_batch = shared_lpips(pred_batch, truth_batch, loss_fn_lpips)
            pce_batch = pce_batch_sum / 2.0
            
            batch_count = pred_batch.shape[0]
            l1_total += l1_batch * batch_count
            mse_total += mse_batch * batch_count
            psnr_total += psnr_batch * batch_count
            ssim_total += ssim_batch * batch_count
            lpips_total += lpips_batch * batch_count
            pce_total += pce_batch * batch_count
            processed += batch_count

            if save_all:
                if save_dir is None:
                    raise ValueError("--save-all requires --save-dir")
                save_dir.mkdir(parents=True, exist_ok=True)
                for local_idx in range(batch_count):
                    sample_id = processed - batch_count + local_idx
                    save_image(batch["start"][local_idx], save_dir / f"{sample_id:05d}_start.png")
                    save_image(batch["inter"][local_idx], save_dir / f"{sample_id:05d}_truth.png")
                    save_image(batch["end"][local_idx], save_dir / f"{sample_id:05d}_end.png")
                    save_image(pred_batch[local_idx], save_dir / f"{sample_id:05d}_pred.png")

            if tracker is not None:
                local_index = processed - batch_count
                source_paths = {}
                dataset = getattr(dataloader, "dataset", None)
                sample_index = (dataset.sample_indices[local_index]
                                if dataset is not None and hasattr(dataset, "sample_indices")
                                else local_index)
                if dataset is not None and hasattr(dataset, "triplets"):
                    triplet = dataset.triplets[local_index]
                    source_paths = {"input_1": str(triplet[0]),
                                    "ground_truth": str(triplet[1]),
                                    "input_2": str(triplet[2])}
                tracker.update(sample_index, psnr_batch, ssim_batch, lpips_batch,
                               pce_batch, pred_batch, truth_batch,
                               batch["start"][:batch_count], batch["end"][:batch_count],
                               source_paths)


            progress.set_postfix(
                samples=processed,
                l1=f"{l1_total / processed:.6f}",
                mse=f"{mse_total / processed:.6f}",
                psnr=f"{psnr_total / processed:.2f}",
                ssim=f"{ssim_total / processed:.4f}",
                lpips=f"{lpips_total / processed:.4f}",
                pce=f"{pce_total / processed:.4f}",
            )

    if processed == 0:
        raise RuntimeError("No samples evaluated.")

    if tracker is not None:
        print("\n" + "="*70)
        print("BEST METRIC IMAGES")
        print("="*70)
        best_rows, best_records = tracker.save_best()
        merge_best_summaries(Path(save_dir).parents[1], best_rows, best_records)
        for metric, entries in tracker.selections().items():
            for rank, entry in enumerate(entries, 1):
                print(f"Best {metric} rank {rank}: sample {entry['sample_index']} = {entry['metrics'][metric]:.6f}")

    return {
        "samples": processed,
        "l1": l1_total / processed,
        "mse": mse_total / processed,
        "psnr": psnr_total / processed,
        "ssim": ssim_total / processed,
        "lpips": lpips_total / processed,
        "pce": pce_total / processed,
    }


def main():
    args = parse_args()
    sample_indices = None
    if args.sample_indices:
        try:
            sample_indices = [int(value) for value in args.sample_indices.split(",")]
        except ValueError as error:
            raise ValueError("--sample-indices must be comma-separated integers") from error
        if not sample_indices or len(sample_indices) != len(set(sample_indices)):
            raise ValueError("--sample-indices must contain unique indices")
        if args.snu_mode == "all":
            raise ValueError("--sample-indices requires one --snu-mode, not 'all'")
        if args.max_samples is not None:
            raise ValueError("--sample-indices cannot be combined with --max-samples")
        args.best_k = len(sample_indices)
    device = resolve_device(args.device)

    if args.dataset_type == "snufilm":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for --dataset-type snufilm")
        if args.batch_size != 1:
            raise ValueError("SNU-FILM publication evaluation requires --batch-size 1")
        root = Path(args.dataset_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SNU-FILM root not found: {root}")
        model, checkpoint_epoch = load_model(args.model_path, device, args.feature_dim)
        results = {}
        for mode in snufilm_modes(args.snu_mode):
            dataset = SNUFILMTriplets(root, mode, args.image_size, sample_indices)
            dataset.print_validation_summary()
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers)
            mode_save_dir = args.save_dir / mode if args.save_dir else None
            try:
                print("SNU-FILM qualitative evaluation")
                print(f"Model: {args.model_name or 'phasenet'}")
                print(f"Mode: {mode}")
                print(f"Resolution: {dataset.input_resolution}")
                print(f"Samples evaluated: {min(len(dataset), args.max_samples or len(dataset))}")
                print(f"Saving only best-k: {args.best_k}")
                results[mode] = evaluate(model, dataloader, device, mode_save_dir,
                                         args.max_samples, args.save_all, args.best_k,
                                         args.model_name or "phasenet", mode)
            except torch.cuda.OutOfMemoryError as error:
                allocated = torch.cuda.memory_allocated(device) / 2**30
                reserved = torch.cuda.memory_reserved(device) / 2**30
                raise RuntimeError(
                    f"CUDA out of memory at {dataset.input_resolution}; "
                    f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB. "
                    "Native SNU-FILM evaluation will not resize or tile automatically."
                ) from error
        model_name = args.model_name or ("PhaseNet-default" if args.feature_dim == 64 else
                                         "PhaseNet-big" if args.feature_dim == 93 else
                                         f"PhaseNet-feature-{args.feature_dim}")
        write_snufilm_results(results, args.metrics_dir, model_name)
        return

    davis_root = resolve_davis_root(args.davis_root, args.dataset_path)
    dataset_path = davis_image_root(davis_root, args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    model, checkpoint_epoch = load_model(args.model_path, device, args.feature_dim)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset_path_str = str(dataset_path)
    dataset_type = args.dataset_type
    if dataset_type == "ucf101" or (dataset_type == "auto" and "ucf101" in dataset_path_str.lower()):
        dataset = UCF101Triplets(dataset_path_str, transform)
    elif dataset_type == "middlebury" or (dataset_type == "auto" and ("middlebury" in dataset_path_str.lower() or "eval-color-allframes" in dataset_path_str.lower())):
        dataset = MiddleburyTriplets(dataset_path_str, transform)
    else:
        train_sequences, val_sequences = load_davis_train_val(davis_root)
        sequences = train_sequences if args.split == "train" else val_sequences
        dataset = Triplets(dataset_path_str, transform, allowed_sequences=sequences)
        print_davis_split(args.split, sequences, len(dataset), evaluation=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Using device: {device}")
    print("Input resolution: 256x256")
    metrics = evaluate(model, dataloader, device, args.save_dir, args.max_samples,
                       args.save_all)
    print("\nEvaluation complete")
    for name in ("samples", "l1", "mse", "psnr", "ssim", "lpips", "pce"):
        print(f"{name.upper()}: {metrics[name]}")


if __name__ == "__main__":
    main()
