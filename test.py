import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, get_input, output_convert

# ============================================================================
# HELPER: Stack pyramid coefficients
# ============================================================================
def stack_pyr_coeffs(coeff_list, batch_size=1):
    """
    Stack pyramid coefficients by flattening each level spatially.
    
    Input: List of 11 tensors, each shape [batch, 2, h_i, w_i]
           (spatial dims vary per level)
    Output: Single tensor [batch, total_flattened_channels]
    """
    flattened = []
    for coeff in coeff_list:
        # coeff shape: [batch, 2, h, w]
        batch = coeff.shape[0]
        # Reshape to [batch, -1] (flatten all dims except batch)
        flat = coeff.view(batch, -1)
        flattened.append(flat)
    
    # Concatenate along channel dimension
    stacked = torch.cat(flattened, dim=1)  # [batch, total_features]
    return stacked


# ============================================================================
# CONFIG
# ============================================================================
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr_type = 1
device = torch.device('cpu')
batch_size = 1
img_size = 256

# ============================================================================
# LOAD DATASET
# ============================================================================
def resolve_dataset_path():
    candidate_paths = [
        Path('/home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p'),
        Path('/home/Salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p'),
        Path(__file__).resolve().parent / 'DAVIS-data' / 'DAVIS' / 'JPEGImages' / '480p',
    ]
    
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    
    raise FileNotFoundError(f"Dataset not found")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

dataset_path = resolve_dataset_path()
dataset = Triplets(dataset_path, transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading model...")
model = torch.load(
    './model/2019-04-15 21_46_19_model.pkl',
    map_location=device,
    weights_only=False
)
model.eval()
print(f"✓ Model loaded: {type(model).__name__}")

# ============================================================================
# INITIALIZE PYRAMID
# ============================================================================
pyr = SCFpyr_PyTorch(
    height=height, nbands=nbands, scale_factor=scale_factor, device=device
)
print(f"✓ Pyramid initialized: height={height}, nbands={nbands}")

# ============================================================================
# PROCESS BATCH
# ============================================================================
print("\n" + "="*70)
print("PROCESSING BATCH")
print("="*70)

dataiter = iter(dataloader)
triplets_batch = next(dataiter)

images_list = [
    torch.stack([
        triplets_batch['start'][i],
        triplets_batch['inter'][i],
        triplets_batch['end'][i]
    ])
    for i in range(batch_size)
]

print(f"Images list length: {len(images_list)}")
print(f"Each image shape: {images_list[0].shape}")

# ============================================================================
# RECONSTRUCT
# ============================================================================
img_recon = np.zeros((img_size, img_size, 3), dtype=np.float32)

with torch.no_grad():
    for channel in range(3):
        print(f"\n--- Channel {channel} ---")
        
        # Build pyramid coefficients
        batch_coeff_list = [
            pyr.build(
                image[:, channel, :, :].unsqueeze(1).to(device),
                pyr_type=pyr_type
            )
            for image in images_list
        ]
        
        # Get input for model
        train_coeff_list, truth_coeff_list = get_input(batch_coeff_list)
        print(f"[DEBUG] train_coeff_list length: {len(train_coeff_list)}")
        print(f"[DEBUG] Shapes: {[c.shape for c in train_coeff_list[:3]]}")
        
        # ✓ Move to device (keep as list)
        train_coeff_list_device = [c.to(device) for c in train_coeff_list]
        
        # ✓ Pass list directly to model
        print(f"Running model inference...")
        pred_coeff = model(train_coeff_list_device)
        print(f"[DEBUG] Model output type: {type(pred_coeff)}")
        print(f"[DEBUG] Model output length: {len(pred_coeff)}")
        print(f"[DEBUG] Output shapes: {[p.shape for p in pred_coeff[:3]]}")
        
        # Convert back to pyramid format and reconstruct
        pred_coeff_converted = output_convert(pred_coeff)
        print(f"[DEBUG] After output_convert: type={type(pred_coeff_converted)}, "
              f"len={len(pred_coeff_converted) if isinstance(pred_coeff_converted, list) else 'N/A'}")
        
        pred_img = pyr.reconstruct(pred_coeff_converted, pyr_type=pyr_type)
        
        # Extract numpy
        pred_np = pred_img[0].detach().cpu().numpy()
        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"  Before normalization: min={pred_np.min():.6f}, max={pred_np.max():.6f}")
        
        # Normalize to [0, 1]
        pred_normalized = pred_np.copy()
        val_min, val_max = pred_normalized.min(), pred_normalized.max()
        
        if val_max > val_min:
            pred_normalized = (pred_normalized - val_min) / (val_max - val_min)
        else:
            print(f"  ⚠ WARNING: Model output is constant")
        
        pred_normalized = np.clip(pred_normalized, 0, 1)
        img_recon[:, :, channel] = pred_normalized
        print(f"  After normalization: min={pred_normalized.min():.6f}, max={pred_normalized.max():.6f}")


# ============================================================================
# VISUALIZATION
# ============================================================================
ground_truth = triplets_batch['inter'].squeeze().numpy().transpose(1, 2, 0)
ground_truth = np.clip(ground_truth, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_recon)
axes[0].set_title('PhaseNet Reconstruction')
axes[0].axis('off')

axes[1].imshow(ground_truth)
axes[1].set_title('Ground Truth')
axes[1].axis('off')

diff = np.abs(img_recon - ground_truth)
im_diff = axes[2].imshow(diff, cmap='hot')
axes[2].set_title(f'Absolute Difference (mean={diff.mean():.4f})')
axes[2].axis('off')
plt.colorbar(im_diff, ax=axes[2])

plt.tight_layout()
plt.savefig('phasenet_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to phasenet_test.png")
plt.show()

# ============================================================================
# COMPUTE METRICS
# ============================================================================
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnr = peak_signal_noise_ratio(ground_truth, img_recon, data_range=1.0)
ssim = structural_similarity(ground_truth, img_recon, data_range=1.0, channel_axis=2)

print("\n" + "="*70)
print("METRICS")
print("="*70)
print(f"PSNR: {psnr:.4f} dB")
print(f"SSIM: {ssim:.4f}")
print("="*70)
