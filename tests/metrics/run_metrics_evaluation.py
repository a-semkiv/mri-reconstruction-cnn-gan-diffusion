import json
from pathlib import Path
import pathlib
import numpy as np

import torch
import torch.serialization
from tqdm import tqdm

torch.serialization.add_safe_globals([pathlib.PosixPath])

from scripts.common.paths import DATA_PATH
from scripts.common.data import build_fastmri_datamodule
from scripts.common.masks import build_mask_r4

from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import UnetModule

from scripts.model_2.gan_module import Pix2PixModule
from scripts.model_3.diffusion_module import DiffusionModule

from tests.metrics import compute_fastmri_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EVAL_SPLIT = "test"   # "val" or "test"

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

    slice_outputs = []

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
                masked_4d = masked.unsqueeze(1)   

                pred = model.sample(
                    masked_4d,
                    num_steps=200,         
                    use_ema=True,
                )

                pred = pred.squeeze(1)  

            else:
                pred = model(masked)

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction shape {pred.shape} does not match target shape {target.shape}"
                )

            pred = pred * std + mean
            target = target * std + mean

            pred_np = pred.cpu().numpy()       
            target_np = target.cpu().numpy()   

            batch_size = pred_np.shape[0]

            for i in range(batch_size):
                fname_i = fnames[i] if isinstance(fnames, (list, tuple)) else fnames
                slice_i = slice_nums[i] if isinstance(slice_nums, torch.Tensor) and slice_nums.ndim > 0 else slice_nums
                maxval_i = max_values[i] if isinstance(max_values, torch.Tensor) and max_values.ndim > 0 else max_values

                slice_outputs.append(
                    {
                        "fname": _to_python_str(fname_i),
                        "slice": _to_python_int(slice_i),
                        "target": target_np[i],
                        "prediction": pred_np[i],
                        "maxval": _to_python_float(maxval_i),
                    }
                )

    metrics_mean, per_slice_metrics = compute_fastmri_metrics(slice_outputs)

    return metrics_mean, per_slice_metrics


def load_model(model_class, checkpoint_path):
    print(f"\nLoading model from {checkpoint_path}")

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=DEVICE,
    )
    model = model.to(DEVICE)
    model.eval()

    return model


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
        raise ValueError(f"Unknown EVAL_SPLIT={EVAL_SPLIT}. Use 'val' or 'test'.")

    print(f"Evaluating split: {EVAL_SPLIT}")
    print("Dataset size:", len(loader.dataset))
    print("Num batches:", len(loader))


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


    results = {}
    results_per_slice = {}

    for model_name, model_info in models.items():
        print(f"Evaluating: {model_name}")

        if model_name == "baseline":
            model = None
        else:
            model = load_model(
                model_info["class"],
                model_info["ckpt"],
            )

        metrics_mean, slice_metrics = evaluate(model, loader)

        results[model_name] = metrics_mean

        for item in slice_metrics:

            key = f"{item['fname']}_{item['slice']}"

            if key not in results_per_slice:
                results_per_slice[key] = {
                    "fname": item["fname"],
                    "slice": item["slice"]
                }

            results_per_slice[key][model_name] = {
                "nmse": item["nmse"],
                "psnr": item["psnr"],
                "ssim": item["ssim"],
            }
        print("\nResults:")
        for k, v in metrics_mean.items():
            print(f"{k}: {v:.6f}")


    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    mean_path = results_dir / f"metrics_mean_{EVAL_SPLIT}_all_models.json"
    per_slice_path = results_dir / f"metrics_per_slice_{EVAL_SPLIT}_all_models.json"

    with open(mean_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(per_slice_path, "w") as f:
        json.dump(list(results_per_slice.values()), f, indent=4)
    
    print("\nResults saved to:")
    print(mean_path)
    print(per_slice_path)


if __name__ == "__main__":
    main()