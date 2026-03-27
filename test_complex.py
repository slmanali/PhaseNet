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

from net.complex_phasenet import ComplexPhaseNet
from net.phasenet import Triplets
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from train_complex import (
    get_complex_input,
    output_convert_complex,
    resolve_dataset_path,
)

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
    model = ComplexPhaseNet(feature_dim=feature_dim).to(device)

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



def evaluate(model, dataloader, device, save_dir=None, max_samples=None):
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
    processed = 0

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
            for channel in range(3):
                batch_coeff_list = [
                    pyr.build(
                        image[:, channel, :, :].unsqueeze(1).to(device),
                        pyr_type=pyr_type,
                    )
                    for image in images_list
                ]

                train_real, train_imag, _, _ = get_complex_input(batch_coeff_list)
                train_real = [tensor.float().to(device) for tensor in train_real]
                train_imag = [tensor.float().to(device) for tensor in train_imag]

                pred_real, pred_imag = model(train_real, train_imag)
                pred_coeff = output_convert_complex(pred_real, pred_imag)
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

            l1_batch = torch.mean(torch.abs(pred_batch - truth_batch)).item()
            mse_batch = torch.mean((pred_batch - truth_batch) ** 2).item()
            psnr_batch = compute_psnr(pred_batch, truth_batch)
            ssim_batch = compute_ssim(pred_batch, truth_batch)

            batch_count = pred_batch.shape[0]
            l1_total += l1_batch * batch_count
            mse_total += mse_batch * batch_count
            psnr_total += psnr_batch * batch_count
            ssim_total += ssim_batch * batch_count
            processed += batch_count

            if save_dir is not None:
                save_batch_visualizations(
                    batch,
                    raw_pred_batch.cpu(),
                    pred_batch.cpu(),
                    save_dir,
                    processed - batch_count,
                )

            progress.set_postfix(
                samples=processed,
                l1=f"{l1_total / processed:.6f}",
                mse=f"{mse_total / processed:.6f}",
                psnr=f"{psnr_total / processed:.2f}",
                ssim=f"{ssim_total / processed:.4f}",
            )

    if processed == 0:
        raise RuntimeError("No samples were evaluated. Check --max-samples and dataset contents.")

    return {
        "samples": processed,
        "l1": l1_total / processed,
        "mse": mse_total / processed,
        "psnr": psnr_total / processed,
        "ssim": ssim_total / processed,
    }


def main():
    args = parse_args()

    device = resolve_device(args.device)
    dataset_path = ensure_dataset_path(args.dataset_path)
    model, checkpoint_epoch = load_model(args.model_path, device, args.feature_dim)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # Choose dataset class based on path or add a new argument
    dataset_path_str = str(dataset_path)
    if "ucf101_interp_ours" in dataset_path_str.lower() or "ucf101" in dataset_path_str.lower():
        print("Using UCF101 (Deep Voxel Flow) dataset structure.")
        dataset = UCF101Triplets(dataset_path_str, transform)
    else:
        print("Using standard Triplets (DAVIS-style) dataset.")
        dataset = Triplets(dataset_path_str, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset triplets: {len(dataset)}")
    print(f"Model path: {args.model_path}")
    if checkpoint_epoch is not None:
        print(f"Checkpoint epoch: {checkpoint_epoch}")

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
    )

    print("\nEvaluation complete")
    print(f"Samples evaluated: {metrics['samples']}")
    print(f"Mean L1:  {metrics['l1']:.6f}")
    print(f"Mean MSE: {metrics['mse']:.6f}")
    print(f"Mean PSNR: {metrics['psnr']:.2f} dB")
    print(f"Mean SSIM: {metrics['ssim']:.4f}")
    if args.save_dir is not None:
        print(f"Saved predictions to: {args.save_dir}")


if __name__ == "__main__":
    main()