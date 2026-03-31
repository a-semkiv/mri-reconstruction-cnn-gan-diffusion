import json
from pathlib import Path
import pathlib
import numpy as np
import matplotlib.pyplot as plt

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
from matplotlib.colors import LinearSegmentedColormap


aqua_hot = LinearSegmentedColormap.from_list(
    "aqua_hot",
    ["black", "teal", "cyan", "lightcyan", "white"]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CASES = {
    "best_case": ("file_brain_AXFLAIR_200_6002483.h5", 5),
    "representative_case": ("file_brain_AXT1PRE_200_6002079.h5", 0),
    "hard_case": ("file_brain_AXFLAIR_203_6000906.h5", 0),
    "model_disagreement_case": ("file_brain_AXFLAIR_209_6001389.h5", 8),
}

OUTPUT_DIR = Path("results/visual_analysis")


def save_img(img, path, cmap="gray"):
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_model(model_class, ckpt):
    model = model_class.load_from_checkpoint(ckpt, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()
    return model


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_func = build_mask_r4()

    transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=True,
    )

    datamodule = build_fastmri_datamodule(
        data_path=DATA_PATH,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        batch_size=1,
        num_workers=4,
        sample_rate=1.0,
    )

    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    print("Loading models...")

    cnn = load_model(
        UnetModule,
        "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_1_full_ds/epoch-epoch=069.ckpt"
    )

    gan = load_model(
        Pix2PixModule,
        "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_2_full_ds/epoch-epoch=071.ckpt"
    )

    diffusion = load_model(
        DiffusionModule,
        "/home/hpc/b192aa/b192aa40/mri_reconstruction/checkpoints/model_3_full_ds/epoch-epoch=098.ckpt"
    )

    for case_name in CASES:
        (OUTPUT_DIR / case_name).mkdir(exist_ok=True)

    print("Searching slices...")

    with torch.inference_mode():

        for batch in tqdm(loader):

            fname = batch.fname[0]
            slice_num = int(batch.slice_num.item())

            for case_name, (target_file, target_slice) in CASES.items():

                if fname == target_file and slice_num == target_slice:

                    print(f"Processing {case_name}")

                    masked = batch.image.to(DEVICE)
                    target = batch.target.to(DEVICE)

                    mean = batch.mean.view(-1,1,1).to(DEVICE)
                    std = batch.std.view(-1,1,1).to(DEVICE)

                    cnn_pred = cnn(masked)

                    gan_pred = gan(masked)

                    diff_pred = diffusion.sample(
                        masked.unsqueeze(1),
                        num_steps=200,
                        use_ema=True,
                    ).squeeze(1)

                    cnn_pred = cnn_pred * std + mean
                    gan_pred = gan_pred * std + mean
                    diff_pred = diff_pred * std + mean
                    target = target * std + mean
                    masked = masked * std + mean

                    cnn_pred = cnn_pred.cpu().numpy()[0]
                    gan_pred = gan_pred.cpu().numpy()[0]
                    diff_pred = diff_pred.cpu().numpy()[0]
                    target = target.cpu().numpy()[0]
                    masked = masked.cpu().numpy()[0]

                    err_cnn = np.abs(cnn_pred - target)
                    err_gan = np.abs(gan_pred - target)
                    err_diff = np.abs(diff_pred - target)

                    case_dir = OUTPUT_DIR / case_name

                    save_img(target, case_dir / "gt.png")
                    save_img(masked, case_dir / "masked.png")

                    save_img(cnn_pred, case_dir / "cnn.png")
                    save_img(gan_pred, case_dir / "gan.png")
                    save_img(diff_pred, case_dir / "diffusion.png")

                    save_img(err_cnn, case_dir / "error_cnn.png", cmap=aqua_hot)
                    save_img(err_gan, case_dir / "error_gan.png", cmap= aqua_hot)
                    save_img(err_diff, case_dir / "error_diffusion.png", cmap= aqua_hot)

    print("Done.")


if __name__ == "__main__":
    main()