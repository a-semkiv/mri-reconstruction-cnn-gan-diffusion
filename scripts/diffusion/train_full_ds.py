from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from fastmri.data.transforms import UnetDataTransform

from scripts.model_3.diffusion_module import DiffusionModule

from scripts.common.seed import seed_everything
from scripts.common.paths import (
    DATA_PATH,
    CHECKPOINTS_DIR,
    TRAIN_LOGS_DIR,
    GENERATIONS_DIR,
    SAVED_MODELS_DIR,
    model_dir,
)

from scripts.common.data import build_fastmri_datamodule_v2
from scripts.common.masks import build_mask_r4
from scripts.common.cleanup import clean_experiment
from scripts.common.io import save_json

from scripts.common.logging.final_ds.diffusion import DiffusionMetricsLoggerFinal
from scripts.common.plotting_callbacks.final_ds.diffusion import (
    DiffusionPlottingCallbackFinal,
)
from scripts.common.generation_callbacks.final_ds.diffusion import (
    DiffusionEpochGenerationCallbackFinal,
)


def main():

    CLEAN_RUN = False
    MAX_EPOCHS = 250

    NUM_DIFFUSION_STEPS = 400
    SAMPLING_STEPS = 200
    VAL_SAMPLING_STEPS = 200

    BATCH_SIZE = 1
    NUM_WORKERS = 4
    LR = 3e-5

    CHANS = 48
    NUM_POOL_LAYERS = 4

    seed_everything(42)

    model_name = "model_3_full_ds"

    ckpt_dir = model_dir(CHECKPOINTS_DIR, model_name)
    log_root = model_dir(TRAIN_LOGS_DIR, model_name)
    logs_dir = log_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = model_dir(GENERATIONS_DIR, model_name)
    ckpt_path = ckpt_dir / "last.ckpt"

    if CLEAN_RUN:
        @rank_zero_only
        def do_cleanup():
            clean_experiment(
                checkpoints_dir=ckpt_dir,
                logs_dir=log_root,
                generations_dir=gen_dir,
            )

        do_cleanup()
        resume_ckpt = None
    else:
        resume_ckpt = ckpt_path if ckpt_path.exists() else None

    print("Checkpoint dir:", ckpt_dir)
    print("Resume ckpt:", resume_ckpt)

    mask_func = build_mask_r4()

    train_transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=False,
    )

    data_module = build_fastmri_datamodule_v2(
        data_path=DATA_PATH,
        train_transform=train_transform,
        test_transform=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sample_rate=1.0,
    )

    data_module.setup("fit")

    model = DiffusionModule(
        lr=LR,
        num_timesteps=NUM_DIFFUSION_STEPS,
        chans=CHANS,
        num_pool_layers=NUM_POOL_LAYERS,
        sampling_steps=SAMPLING_STEPS,
        val_sampling_steps=VAL_SAMPLING_STEPS,
    )

    logger = TensorBoardLogger(
        save_dir=log_root,
        name="conditional_ddpm_r4_full_ds",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:03d}",
        save_last=True,
        save_top_k=-1,
    )

    metrics_cb = DiffusionMetricsLoggerFinal(save_dir=logs_dir)

    plot_cb = DiffusionPlottingCallbackFinal(
        metrics_path=logs_dir / "metrics.json",
        save_dir=log_root / "plots",
        model_name="Conditional Diffusion (R=4) full dataset",
    )

    monitor_loader = data_module.val_dataloader()
    fixed_batch = next(iter(monitor_loader))

    epoch_gen_cb = DiffusionEpochGenerationCallbackFinal(
        batch=fixed_batch,
        save_dir=gen_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_steps=VAL_SAMPLING_STEPS,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=3,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        ),
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            metrics_cb,
            plot_cb,
            epoch_gen_cb,
        ],
        log_every_n_steps=50,
        gradient_clip_val=0.5,
        accumulate_grad_batches=4,
        precision=32,
        deterministic=True,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_ckpt,
    )

    train_summary = {
        "model": "Conditional Diffusion",
        "task": "MRI Reconstruction",
        "dataset": "train + val - monitor",
        "acceleration": 4,
        "epochs_trained": trainer.current_epoch + 1,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "num_diffusion_steps": NUM_DIFFUSION_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "val_sampling_steps": VAL_SAMPLING_STEPS,
        "chans": CHANS,
        "num_pool_layers": NUM_POOL_LAYERS,
        "devices": 3,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 0.5,
    }

    save_json(train_summary, log_root / "train_stats.json")
    print(f"\nTraining stats saved to {log_root / 'train_stats.json'}")

    saved_model_dir = model_dir(SAVED_MODELS_DIR, model_name)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu")
    model_cpu.eval()

    weights_path = saved_model_dir / "model_weights.pt"
    torch.save(model_cpu.state_dict(), weights_path)

    model_config = {
        "model_class": "DiffusionModule",
        "lr": LR,
        "num_timesteps": NUM_DIFFUSION_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "val_sampling_steps": VAL_SAMPLING_STEPS,
        "chans": CHANS,
        "num_pool_layers": NUM_POOL_LAYERS,
        "acceleration": 4,
        "mask": "random_r4",
        "dataset": "train + val - monitor",
    }

    save_json(model_config, saved_model_dir / "model_config.json")

    print(f"Model weights saved to {weights_path}")
    print(f"Model config saved to {saved_model_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()