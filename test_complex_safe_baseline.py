"""
Evaluation script for Complex-Valued PhaseNet checkpoints.

Loads either a final ``state_dict`` saved by ``train_complex.py`` or an
intermediate checkpoint containing ``model_state_dict`` and evaluates it on
triplets generated from the DAVIS-style dataset layout used by this repo.
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
from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe
from net.phasenet import Triplets
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from train_complex_safe_baseline import (
    get_complex_input,
    output_convert_complex,
    resolve_dataset_path,
)
from utils.davis import (davis_image_root, load_davis_train_val,
                         print_davis_split, resolve_davis_root)
from utils.metrics import (compute_l1 as shared_l1, compute_mse as shared_mse,
                           compute_psnr as shared_psnr, compute_ssim as shared_ssim,
                           compute_lpips as shared_lpips, compute_complex_pce)
from utils.snufilm import (SNUFILMTriplets, SNU_MODES, snufilm_modes,
                           write_snufilm_results)

class UCF101Triplets(torch.utils.data.Dataset):
    """Dataset for ucf101_interp_ours structure from Deep Voxel Flow paper."""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # Find all subfolders that contain the required frames
        self.samples = []
        for item in sorted(self.root.iterdir()):
            if item.is_dir():
                if (item / "frame_00.png").exists() and \
                   (item / "frame_02.png").exists() and \
                   (item / "frame_01_gt.png").exists():
                    self.samples.append(item)
        
        print(f"Found {len(self.samples)} UCF101 triplets.")
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No valid triplets found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        
        # Read images directly to tensor (range [0, 1]) — do NOT apply ToTensor again
        start = torchvision.io.read_image(str(folder / "frame_00.png")).float() / 255.0
        end   = torchvision.io.read_image(str(folder / "frame_02.png")).float() / 255.0
        inter = torchvision.io.read_image(str(folder / "frame_01_gt.png")).float() / 255.0
        
        # Apply only Resize (and any future transforms that support tensors)
        if self.transform:
            # Remove ToTensor from the transform for UCF101 (it is already a tensor)
            # We keep Resize only
            resize_only = transforms.Compose([t for t in self.transform.transforms if not isinstance(t, transforms.ToTensor)])
            start = resize_only(start)
            end   = resize_only(end)
            inter = resize_only(inter)
        
        return {
            "start": start,
            "end": end,
            "inter": inter,
        }


class MiddleburyTriplets(torch.utils.data.Dataset):
    """Dataset for Middlebury eval-color-allframes structure."""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # The real data is under eval-data/
        data_root = self.root / "eval-data"
        if not data_root.exists():
            data_root = self.root  # fallback if user points directly to eval-data
        
        self.samples = []
        for item in sorted(data_root.iterdir()):
            if item.is_dir():
                f10 = item / "frame10.png"
                f12 = item / "frame12.png"
                f11 = item / "frame11.png"
                if f10.exists() and f12.exists() and f11.exists():
                    self.samples.append(item)
        
        print(f"Found {len(self.samples)} Middlebury sequences.")
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No valid sequences found in {data_root}")

    def __len__(self):
        return len(self.samples)

    def _to_rgb(self, img):
        """Ensure image is 3-channel RGB (repeat grayscale if needed)."""
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:  # RGBA
            img = img[:3]
        return img

    def __getitem__(self, idx):
        folder = self.samples[idx]
        
        start = torchvision.io.read_image(str(folder / "frame10.png")).float() / 255.0
        end   = torchvision.io.read_image(str(folder / "frame12.png")).float() / 255.0
        inter = torchvision.io.read_image(str(folder / "frame11.png")).float() / 255.0
        
        # Force RGB
        start = self._to_rgb(start)
        end   = self._to_rgb(end)
        inter = self._to_rgb(inter)
        
        if self.transform:
            resize_only = transforms.Compose([t for t in self.transform.transforms 
                                            if not isinstance(t, transforms.ToTensor)])
            start = resize_only(start)
            end   = resize_only(end)
            inter = resize_only(inter)
        
        return {
            "start": start,
            "end": end,
            "inter": inter,
        }


class BestMetricsTracker:
    """Track and save images with best metrics."""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best = {
            "psnr": {"value": -float("inf"), "idx": None, "pred": None, "truth": None},
            "ssim": {"value": -float("inf"), "idx": None, "pred": None, "truth": None},
            "lpips": {"value": float("inf"), "idx": None, "pred": None, "truth": None},
            "pce": {"value": float("inf"), "idx": None, "pred": None, "truth": None},
        }
    def update(self, batch_idx, psnr, ssim, lpips_val, pce_val, pred, truth):
        """Update best metrics if current is better."""
        if psnr > self.best["psnr"]["value"]:
            self.best["psnr"]["value"] = psnr
            self.best["psnr"]["idx"] = batch_idx
            self.best["psnr"]["pred"] = pred.clone().detach().cpu()
            self.best["psnr"]["truth"] = truth.clone().detach().cpu()
        if ssim > self.best["ssim"]["value"]:
            self.best["ssim"]["value"] = ssim
            self.best["ssim"]["idx"] = batch_idx
            self.best["ssim"]["pred"] = pred.clone().detach().cpu()
            self.best["ssim"]["truth"] = truth.clone().detach().cpu()
        if lpips_val < self.best["lpips"]["value"]:
            self.best["lpips"]["value"] = lpips_val
            self.best["lpips"]["idx"] = batch_idx
            self.best["lpips"]["pred"] = pred.clone().detach().cpu()
            self.best["lpips"]["truth"] = truth.clone().detach().cpu()
        if pce_val < self.best["pce"]["value"]:
            self.best["pce"]["value"] = pce_val
            self.best["pce"]["idx"] = batch_idx
            self.best["pce"]["pred"] = pred.clone().detach().cpu()
            self.best["pce"]["truth"] = truth.clone().detach().cpu()

    def save_best(self):
        """Save best images for each metric."""
        for metric_name, data in self.best.items():
            if data["pred"] is None:
                print(f"⚠ No data for {metric_name}")
                continue
            
            metric_dir = self.save_dir / metric_name
            metric_dir.mkdir(exist_ok=True)
            
            # Prepare pred (normalized for visualization)
            pred_normalized = normalize_for_visualization(
                data["pred"].squeeze(0) if data["pred"].dim() == 4 else data["pred"]
            )
            
            # Prepare truth
            truth_normalized = data["truth"].squeeze(0) if data["truth"].dim() == 4 else data["truth"]
            
            # Concatenate side-by-side (Truth on left, Prediction on right)
            comparison = torch.cat([truth_normalized, pred_normalized], dim=2)
            
            # Save the combined image
            save_image(comparison, metric_dir / f"best_{metric_name}_comparison.png")
            
            # Save metric value
            with open(metric_dir / "metric_value.txt", "w") as f:
                f.write(f"{data['value']:.6f}\n")
                f.write(f"Batch index: {data['idx']}\n")
            
            # Determine unit
            if metric_name == "pce":
                unit = f"{data['value']:.4f} rad ({data['value'] * 180.0 / math.pi:.2f}°)"
            else:
                unit = f"{data['value']:.6f}"
            print(f"✓ Saved best {metric_name:6s}: {data['value']:.6f} (batch {data['idx']})")
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Complex PhaseNet model on triplet data."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a *_complex_final.pth or checkpoint *.pth file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Path to the DAVIS-style dataset root. If omitted, the same dataset "
            "resolution logic as train_complex.py is used."
        ),
    )
    parser.add_argument("--davis-root", type=str, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dataset-type", choices=("auto", "davis", "ucf101", "middlebury", "snufilm"), default="auto")
    parser.add_argument("--snu-mode", choices=(*SNU_MODES, "all"), default="easy")
    parser.add_argument("--image-size", choices=("native", "256"), default="256",
                        help="SNU-FILM input resolution; native is not resized.")
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help=("Model inference tile edge. Native SNU-FILM defaults to 256 to "
              "reduce peak CUDA memory; pass 0 to disable tiling."),
    )
    parser.add_argument("--model-name", default="ComplexPhaseNet-full",
                        help="Paper CSV model label.")
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
        help="Optional directory where predicted/ground-truth/start/end frames are saved.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=32,
        help="Feature dimension used when the model was trained.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device, feature_dim):
    checkpoint = torch.load(model_path, map_location=device)
    model = ComplexPhaseNetSafe(feature_dim=feature_dim).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch")
    else:
        model.load_state_dict(checkpoint)
        epoch = None

    model.eval()
    return model, epoch


def ensure_dataset_path(dataset_path_arg):
    dataset_path = dataset_path_arg or resolve_dataset_path()
    dataset_path = Path(dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}. "
            "Pass --dataset-path or set PHASENET_DATASET_PATH."
        )
    return dataset_path


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)

def compute_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """
    Compute mean SSIM between two batches of images (pure PyTorch, no external dependencies).
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

def normalize_for_visualization(image):
    """
    Convert a single image tensor to a viewable [0, 1] range.

    PhaseNet reconstructions can legitimately fall outside [0, 1] or occupy only
    a tiny value range. Saving them with a hard clamp can therefore produce
    nearly-black debug images even when the reconstruction contains structure.
    This helper preserves the raw tensor for metrics while stretching the saved
    visualization to the full display range.
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

def compute_lpips(pred, target, net_type='alex', device='cuda'):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Args:
        pred, target: [B, 3, H, W] in [0, 1]
        net_type: 'alex', 'vgg', or 'squeeze'
    
    Returns:
        Mean LPIPS distance (0=identical, 1=maximally different)
    """
    loss_fn = lpips.LPIPS(net=net_type, verbose=False).to(device)
    
    # lpips expects [-1, 1] range
    pred_normalized = 2 * pred - 1
    target_normalized = 2 * target - 1
    
    with torch.no_grad():
        distance = loss_fn(pred_normalized, target_normalized)
    
    return distance.mean().item()

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

def save_batch_visualizations(batch, raw_predictions, clamped_predictions, save_dir, sample_offset):
    save_dir.mkdir(parents=True, exist_ok=True)
    for batch_idx in range(raw_predictions.shape[0]):
        sample_id = sample_offset + batch_idx
        save_image(batch["start"][batch_idx], save_dir / f"{sample_id:05d}_start.png")
        save_image(batch["end"][batch_idx], save_dir / f"{sample_id:05d}_end.png")
        save_image(batch["inter"][batch_idx], save_dir / f"{sample_id:05d}_truth.png")
        save_image(
            clamped_predictions[batch_idx],
            save_dir / f"{sample_id:05d}_pred_raw.png",
        )
        save_image(
            normalize_for_visualization(raw_predictions[batch_idx]),
            save_dir / f"{sample_id:05d}_pred.png",
        )

def evaluate(model, dataloader, device, save_dir=None, max_samples=None, tile_size=None):
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

    tracker = BestMetricsTracker(save_dir) if save_dir else None
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
                outputs = get_complex_input(batch_coeff_list)
                if len(outputs) == 6:
                    train_real, train_imag, truth_real, truth_imag, hp_start, hp_end = outputs
                    hp_start = hp_start.float().to(device)
                    hp_end = hp_end.float().to(device)
                    hp_mid = 0.5 * (hp_start + hp_end)
                else:
                    train_real, train_imag, truth_real, truth_imag = outputs
                    hp_mid = None
                train_real = [tensor.float().to(device) for tensor in train_real]
                train_imag = [tensor.float().to(device) for tensor in train_imag]
                truth_real = [tensor.float().to(device) for tensor in truth_real]
                truth_imag = [tensor.float().to(device) for tensor in truth_imag]

                pred_real, pred_imag = model(train_real, train_imag, tile_size=tile_size)
                pce_channel = compute_complex_pce(truth_real, truth_imag, pred_real, pred_imag)
                pce_batch_sum += pce_channel

                pred_coeff = output_convert_complex(pred_real, pred_imag, highpass=hp_mid)
                pred_img = pyr.reconstruct(pred_coeff, pyr_type=pyr_type)
                recon_channels.append(pred_img.unsqueeze(1))

            raw_pred_batch = torch.cat(recon_channels, dim=1)
            pred_batch = raw_pred_batch.clamp(0, 1)
            truth_batch = batch["inter"].to(device)

            if max_samples is not None:
                remaining = max_samples - processed
                pred_batch = pred_batch[:remaining]
                truth_batch = truth_batch[:remaining]
                batch = {key: value[:remaining] for key, value in batch.items()}

            l1_batch = shared_l1(pred_batch, truth_batch)
            mse_batch = shared_mse(pred_batch, truth_batch)
            psnr_batch = shared_psnr(pred_batch, truth_batch)
            ssim_batch = shared_ssim(pred_batch, truth_batch)
            lpips_batch = shared_lpips(pred_batch, truth_batch, loss_fn_lpips)
            pce_batch = pce_batch_sum / 3.0

            batch_count = pred_batch.shape[0]
            l1_total += l1_batch * batch_count
            mse_total += mse_batch * batch_count
            psnr_total += psnr_batch * batch_count
            ssim_total += ssim_batch * batch_count
            lpips_total += lpips_batch * batch_count
            pce_total += pce_batch * batch_count
            processed += batch_count

            # ✓ Update tracker
            if tracker is not None:
                tracker.update(
                    batch_size,
                    psnr_batch,
                    ssim_batch,
                    lpips_batch,
                    pce_batch,
                    pred_batch,
                    truth_batch
                )

            # if save_dir is not None:
            #     save_batch_visualizations(
            #         batch,
            #         raw_pred_batch.cpu(),
            #         pred_batch.cpu(),
            #         save_dir,
            #         processed - batch_count,
            #     )

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
        raise RuntimeError("No samples were evaluated. Check --max-samples and dataset contents.")
    # ✓ Save best images
    if tracker is not None:
        print("\n" + "="*70)
        print("BEST METRIC IMAGES")
        print("="*70)
        tracker.save_best()
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
    if args.tile_size is not None and args.tile_size < 0:
        raise ValueError("--tile-size must be non-negative")
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
        tile_size = 256 if args.tile_size is None and args.image_size == "native" else args.tile_size
        if tile_size:
            print(f"Model inference tile size: {tile_size}x{tile_size}")
        results = {}
        for mode in snufilm_modes(args.snu_mode):
            dataset = SNUFILMTriplets(root, mode, args.image_size)
            dataset.print_validation_summary()
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers)
            mode_save_dir = args.save_dir / mode if args.save_dir else None
            try:
                results[mode] = evaluate(model, dataloader, device, mode_save_dir,
                                         args.max_samples, tile_size)
            except torch.cuda.OutOfMemoryError as error:
                allocated = torch.cuda.memory_allocated(device) / 2**30
                reserved = torch.cuda.memory_reserved(device) / 2**30
                raise RuntimeError(
                    f"CUDA out of memory at {dataset.input_resolution}; "
                    f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB. "
                    "Try a smaller --tile-size (for example, 128)."
                ) from error
        model_name = args.model_name
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
    metrics = evaluate(model, dataloader, device, args.save_dir, args.max_samples)
    print("\nEvaluation complete")
    for name in ("samples", "l1", "mse", "psnr", "ssim", "lpips", "pce"):
        print(f"{name.upper()}: {metrics[name]}")


if __name__ == "__main__":
    main()
