from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from fastmri.data.transforms import UnetDataTransform

from fastmri.pl_modules import UnetModule

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
from scripts.common.profiling import TrainingProfiler
from scripts.common.io import save_json
from scripts.common.cleanup import clean_experiment

from scripts.common.logging.final_ds.cnn import CNNMetricsLoggerFinal
from scripts.common.generation_callbacks.final_ds.cnn import (
    CNNEpochGenerationCallbackFinal
)
from scripts.common.plotting_callbacks.final_ds.cnn import CNNPlottingCallbackFinal


def main():

    CLEAN_RUN = False
    MAX_EPOCHS = 70
    SAMPLE_RATE = 1.0

    seed_everything(42)

    model_name = "model_1_full_ds"

    ckpt_dir = model_dir(CHECKPOINTS_DIR, model_name)
    log_root = model_dir(TRAIN_LOGS_DIR, model_name)

    logs_dir = log_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = model_dir(GENERATIONS_DIR, model_name)

    ckpt_path = ckpt_dir / "last.ckpt"

    if CLEAN_RUN:
        print("CLEAN_RUN=True -> starting fresh experiment")

        clean_experiment(
            checkpoints_dir=ckpt_dir,
            logs_dir=log_root,
            generations_dir=gen_dir,
        )

        resume_ckpt = None

    else:
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            resume_ckpt = ckpt_path
        else:
            print("No checkpoint found -> starting fresh training")
            resume_ckpt = None

    print(f"Resume checkpoint: {resume_ckpt}")

    mask_func = build_mask_r4()

    train_transform = UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=False,
    )

    data_module = build_fastmri_datamodule_v2(
        data_path=DATA_PATH,
        train_transform=train_transform,
        batch_size=1,
        num_workers=8,
        sample_rate=SAMPLE_RATE,
    )

    data_module.setup("fit")

    model = UnetModule(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=3e-4,
    )

    logger = TensorBoardLogger(
        save_dir=log_root,
        name="unet_r4_full_ds",
    )


    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:03d}",
        save_top_k=-1,
        save_last=True,
    )

    metrics_cb = CNNMetricsLoggerFinal(save_dir=logs_dir)

    plot_cb = CNNPlottingCallbackFinal(
        metrics_path=logs_dir / "metrics.json",
        save_dir=log_root / "plots",
        model_name="U-Net (R=4) full dataset",
    )

    monitor_loader = data_module.val_dataloader()
    fixed_batch = next(iter(monitor_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_gen_cb = CNNEpochGenerationCallbackFinal(
        batch=fixed_batch,
        save_dir=gen_dir,
        device=device,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            metrics_cb,
            plot_cb,
            epoch_gen_cb,
        ],
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    profiler = TrainingProfiler()
    profiler.start()

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_ckpt,
    )

    profiler.stop()

    profiler.set_training_shape(
        epochs=trainer.current_epoch + 1,
        batches=trainer.num_training_batches,
    )

    stats = profiler.get_stats()

    train_summary = {
        "model": "U-Net (fastMRI baseline)",
        "task": "MRI Reconstruction",
        "dataset": "train + val - monitor",
        "acceleration": 4,
        "epochs_trained": trainer.current_epoch + 1,
        "batch_size": data_module.batch_size,
        "learning_rate": model.lr,
        **stats,
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
        "model_class": "UnetModule",
        "in_chans": 1,
        "out_chans": 1,
        "chans": 32,
        "num_pool_layers": 4,
        "drop_prob": 0.0,
        "lr": 3e-4,
        "acceleration": 4,
        "mask": "random_r4",
        "dataset": "train + val - monitor",
    }

    save_json(model_config, saved_model_dir / "model_config.json")

    print(f"Model weights saved to {weights_path}")
    print(f"Model config saved to {saved_model_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()