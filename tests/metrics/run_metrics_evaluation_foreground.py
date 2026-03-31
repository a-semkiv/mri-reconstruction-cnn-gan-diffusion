import json
from pathlib import Path
import pathlib
import numpy as np

import torch
import torch.serialization
from tqdm import tqdm

from scipy.ndimage import binary_fill_holes, label, binary_dilation
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

torch.serialization.add_safe_globals([pathlib.PosixPath])

from scripts.common.paths import DATA_PATH
from scripts.common.data import build_fastmri_datamodule
from scripts.common.masks import build_mask_r4

from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import UnetModule

from scripts.model_2.gan_module import Pix2PixModule
from scripts.model_3.diffusion_module import DiffusionModule


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SPLIT = "test"

def build_head_mask(img):
    img_norm = img - img.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()

    try:
        t = threshold_otsu(img_norm)
        mask = img_norm > (0.5 * t)
    except Exception:
        mask = img_norm > 0.02

    labeled, num = label(mask)
    if num > 0:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    mask = binary_fill_holes(mask)

    mask = binary_dilation(mask, iterations=8)

    return mask.astype(np.float32)


def nmse_fastmri(gt, pred, mask):
    diff = (gt - pred) * mask
    return np.linalg.norm(diff) ** 2 / (np.linalg.norm(gt * mask) ** 2 + 1e-8)


def psnr_fastmri(gt, pred, mask, maxval):
    gt_masked = gt[mask > 0]
    pred_masked = pred[mask > 0]

    return peak_signal_noise_ratio(
        gt_masked,
        pred_masked,
        data_range=maxval,
    )


def ssim_fastmri(gt, pred, mask, maxval):
    coords = np.where(mask > 0)

    if coords[0].size == 0 or coords[1].size == 0:
        return np.nan

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    gt_crop = gt[y_min:y_max + 1, x_min:x_max + 1]
    pred_crop = pred[y_min:y_max + 1, x_min:x_max + 1]

    return structural_similarity(
        gt_crop,
        pred_crop,
        data_range=maxval,
    )

def _to_python_int(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return int(x.item())
        return int(x[0].item())
    return int(x)


def _to_python_float(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return float(x.item())
        return float(x[0].item())
    return float(x)


def _to_python_str(x):
    if isinstance(x, (list, tuple)):
        return str(x[0])
    return str(x)


def evaluate(model, loader):

    if model is not None:
        model.eval()

    roi_per_slice = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):

            masked = batch.image.to(DEVICE, non_blocking=True)
            target = batch.target.to(DEVICE, non_blocking=True)

            mean = batch.mean.view(-1, 1, 1).to(DEVICE, non_blocking=True)
            std = batch.std.view(-1, 1, 1).to(DEVICE, non_blocking=True)

            fnames = batch.fname
            slice_nums = batch.slice_num
            max_values = batch.max_value

            if model is None:
                pred = masked

            elif isinstance(model, DiffusionModule):
                pred = model.sample(
                    masked.unsqueeze(1),
                    num_steps=10,
                    use_ema=True,
                ).squeeze(1)

            else:
                pred = model(masked)

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction shape {pred.shape} != target shape {target.shape}"
                )

            pred = pred * std + mean
            target = target * std + mean

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            B = pred_np.shape[0]

            for i in range(B):
                fname = _to_python_str(fnames[i])
                slice_i = _to_python_int(slice_nums[i])
                maxval = _to_python_float(max_values[i])

                gt = target_np[i]
                pr = pred_np[i]

                mask = build_head_mask(gt)

                roi_per_slice.append(
                    {
                        "fname": fname,
                        "slice": slice_i,
                        "nmse": float(nmse_fastmri(gt, pr, mask)),
                        "psnr": float(psnr_fastmri(gt, pr, mask, maxval)),
                        "ssim": float(ssim_fastmri(gt, pr, mask, maxval)),
                    }
                )

    valid_nmse = [x["nmse"] for x in roi_per_slice if np.isfinite(x["nmse"])]
    valid_psnr = [x["psnr"] for x in roi_per_slice if np.isfinite(x["psnr"])]
    valid_ssim = [x["ssim"] for x in roi_per_slice if np.isfinite(x["ssim"])]

    roi_mean = {
        "nmse": float(np.mean(valid_nmse)),
        "psnr": float(np.mean(valid_psnr)),
        "ssim": float(np.mean(valid_ssim)),
    }

    return roi_mean, roi_per_slice


def load_model(model_class, checkpoint_path):
    print(f"\nLoading model from {checkpoint_path}")

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=DEVICE,
    )

    return model.to(DEVICE).eval()


def main():

    print("\nPreparing dataloader...")

    torch.backends.cudnn.benchmark = True

    mask_func = build_mask_r4()

    train_transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=False,
    )

    val_transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=True,
    )

    test_transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=True,
    )

    datamodule = build_fastmri_datamodule(
        data_path=DATA_PATH,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=1,
        num_workers=4,
        sample_rate=1.0,
    )

    if EVAL_SPLIT == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    elif EVAL_SPLIT == "test":
        datamodule.setup("test")
        loader = datamodule.test_dataloader()
    else:
        raise ValueError("EVAL_SPLIT must be 'val' or 'test'")

    print(f"Evaluating split: {EVAL_SPLIT}")
    print("Dataset size:", len(loader.dataset))

    models = {
        "baseline": None,
        "cnn": {
            "class": UnetModule,
            "ckpt": "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_1_full_ds/epoch-epoch=069.ckpt",
        },
        "gan": {
            "class": Pix2PixModule,
            "ckpt": "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_2_full_ds/epoch-epoch=071.ckpt",
        },
        "diffusion": {
            "class": DiffusionModule,
            "ckpt": "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_3_full_ds/epoch-epoch=098.ckpt",
        },
    }

    results_mean = {}
    results_per_slice = {}

    for model_name, model_info in models.items():

        print(f"Evaluating: {model_name}")

        model = None if model_name == "baseline" else load_model(
            model_info["class"], model_info["ckpt"]
        )

        roi_mean, roi_slices = evaluate(model, loader)

        results_mean[model_name] = roi_mean

        for item in roi_slices:
            key = f"{item['fname']}_{item['slice']}"

            if key not in results_per_slice:
                results_per_slice[key] = {
                    "fname": item["fname"],
                    "slice": item["slice"],
                }

            results_per_slice[key][model_name] = {
                "nmse": item["nmse"],
                "psnr": item["psnr"],
                "ssim": item["ssim"],
            }

        print("\nROI metrics:")
        for k, v in roi_mean.items():
            print(f"{k}: {v:.6f}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    mean_path = results_dir / f"metrics_mean_{EVAL_SPLIT}_all_models.json"
    per_slice_path = results_dir / f"metrics_per_slice_{EVAL_SPLIT}_all_models.json"

    with open(mean_path, "w") as f:
        json.dump(results_mean, f, indent=4)

    with open(per_slice_path, "w") as f:
        json.dump(list(results_per_slice.values()), f, indent=4)

    print("\nSaved to:")
    print(mean_path)
    print(per_slice_path)


if __name__ == "__main__":
    main()