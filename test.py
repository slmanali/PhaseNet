import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, show_Triplets_batch, get_input,output_convert


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

def ensure_dataset_path():
    dataset_path = resolve_dataset_path()
    dataset_path = Path(dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}. "
            "Pass --dataset-path or set PHASENET_DATASET_PATH."
        )
    return dataset_path

# Load dataset
batch_size = 1
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

dataset_path = ensure_dataset_path()
dataset = Triplets(str(dataset_path), transform)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

dataiter = iter(trainloader)
Triplets_batch = next(dataiter)
# show_Triplets_batch(Triplets_batch)
# define para
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr_type = 1
device = torch.device('cpu')# cuda:0
pyr = SCFpyr_PyTorch(height=height, nbands=nbands, scale_factor=scale_factor,device=device)


# load model
model = torch.load(
    './model/2019-04-15 21:46:19_model.pkl',
    map_location=device,
    weights_only=False
)
# model.load_state_dict(state_dict)
model.eval()
# get images_list [[N,C,H,W],[N,C,H,W],...] len(images_list)=batch_size
images_list = [torch.stack([Triplets_batch['start'][i],
                            Triplets_batch['inter'][i],
                            Triplets_batch['end'][i]]) for i in range(batch_size)]

img_recon = np.empty((256, 256, 3), dtype=np.float32)

plt.figure(1, figsize=(12, 4))
with torch.no_grad():
    for channel in range(3):
        batch_coeff_list = [
            pyr.build(image[:, channel, :, :].unsqueeze(1).to(device), pyr_type=pyr_type)
            for image in images_list
        ]
        train_coeff, truth_coeff = get_input(batch_coeff_list)
        truth_from_coeff = pyr.reconstruct(output_convert(truth_coeff), pyr_type=pyr_type)
        truth_coeff_img = truth_from_coeff[0].detach().cpu().numpy()
        truth_coeff_img = np.nan_to_num(truth_coeff_img)

        print(
            f"truth_from_coeff ch{channel}: "
            f"min={truth_coeff_img.min():.8f}, "
            f"max={truth_coeff_img.max():.8f}, "
            f"mean={truth_coeff_img.mean():.8f}"
        )
        pre_coeff = model(train_coeff)

        pre_img = pyr.reconstruct(output_convert(pre_coeff), pyr_type=pyr_type)

        img = pre_img[0].detach().cpu().numpy()
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        print(
            f"channel {channel}: "
            f"min={img.min():.8f}, max={img.max():.8f}, mean={img.mean():.8f}"
        )

        img_recon[:, :, channel] = img

        disp = img - img.min()
        if disp.max() > 0:
            disp = disp / disp.max()

        plt.subplot(1, 3, channel + 1)
        plt.imshow(disp, cmap='gray')
        plt.title(f'channel {channel}')
        plt.axis('off')

truth = Triplets_batch['inter'].squeeze().numpy().transpose(1, 2, 0)

print(
    f"img_recon overall: "
    f"min={img_recon.min():.8f}, max={img_recon.max():.8f}, mean={img_recon.mean():.8f}"
)

# raw display
raw_vis = np.clip(img_recon, 0.0, 1.0)

# normalized RGB display only for diagnosis
norm_vis = img_recon - img_recon.min()
if norm_vis.max() > 0:
    norm_vis = norm_vis / norm_vis.max()

truth_vis = np.clip(truth, 0.0, 1.0)

plt.figure(2, figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(raw_vis)
plt.title('raw/clipped recon')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(norm_vis)
plt.title('normalized recon')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(truth_vis)
plt.title('ground truth')
plt.axis('off')

plt.tight_layout()
plt.savefig('phasenet_debug.png', dpi=200, bbox_inches='tight')
print('Saved figure to phasenet_debug.png')