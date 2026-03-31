import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH = 64

CASES = {
    "representative_case": ("file_brain_AXT1PRE_200_6002079.h5", 0),
    "model_disagreement_case": ("file_brain_AXFLAIR_209_6001389.h5", 8),
}

OUTPUT_DIR = Path("results/zoom_analysis")



def save_img(img, path, cmap="gray"):

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_roi_image(img, x, y, path):

    fig, ax = plt.subplots(figsize=(4,4))

    ax.imshow(img, cmap="gray")

    rect = patches.Rectangle(
        (x, y),
        PATCH,
        PATCH,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def crop(img, x, y):
    return img[y:y+PATCH, x:x+PATCH]


def find_error_roi(gt, cnn, gan, diff):

    error = np.abs(cnn - gt) + np.abs(gan - gt) + np.abs(diff - gt)

    mask = gt > 0.05 * gt.max()
    error = error * mask

    h, w = error.shape
    best_score = -1
    best_xy = (0,0)

    step = PATCH // 4

    for y in range(0, h-PATCH, step):
        for x in range(0, w-PATCH, step):

            region = error[y:y+PATCH, x:x+PATCH]
            score = region.mean()

            if score > best_score:
                best_score = score
                best_xy = (x,y)

    return best_xy


def find_disagreement_roi(cnn, gan, diff):

    stack = np.stack([cnn, gan, diff], axis=0)
    disagreement = np.std(stack, axis=0)

    h, w = disagreement.shape
    best_score = -1
    best_xy = (0,0)

    step = PATCH // 4

    for y in range(0, h-PATCH, step):
        for x in range(0, w-PATCH, step):

            region = disagreement[y:y+PATCH, x:x+PATCH]
            score = region.mean()

            if score > best_score:
                best_score = score
                best_xy = (x,y)

    return best_xy


def load_model(model_class, ckpt):

    model = model_class.load_from_checkpoint(
        ckpt,
        map_location=DEVICE
    )

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

                    cnn_pred = cnn_pred.cpu().numpy()[0]
                    gan_pred = gan_pred.cpu().numpy()[0]
                    diff_pred = diff_pred.cpu().numpy()[0]
                    target = target.cpu().numpy()[0]


                    if case_name == "representative_case":
                        x, y = find_error_roi(
                            target,
                            cnn_pred,
                            gan_pred,
                            diff_pred
                        )

                    else:
                        x, y = find_error_roi(
                            target,
                            cnn_pred,
                            gan_pred,
                            diff_pred
                        )

                    case_dir = OUTPUT_DIR / case_name

                    save_roi_image(
                        target,
                        x,
                        y,
                        case_dir / "roi_full.png"
                    )

                    save_img(crop(target, x, y), case_dir / "zoom_gt.png")
                    save_img(crop(cnn_pred, x, y), case_dir / "zoom_cnn.png")
                    save_img(crop(gan_pred, x, y), case_dir / "zoom_gan.png")
                    save_img(crop(diff_pred, x, y), case_dir / "zoom_diffusion.png")

    print("Zoom analysis complete.")


if __name__ == "__main__":
    main()