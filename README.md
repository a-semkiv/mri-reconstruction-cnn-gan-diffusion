# MRI Reconstruction: CNN vs. GAN vs. Diffusion

Code for my bachelor's thesis on accelerated brain MRI reconstruction. The project compares three learned priors for image-domain reconstruction of undersampled k-space data on the [fastMRI](https://fastmri.med.nyu.edu/) brain dataset:

- **CNN**: U-Net trained with an L1 loss (deterministic regression)
- **GAN**: Pix2Pix with the same U-Net as generator and a PatchGAN discriminator (adversarial + L1 loss)
- **Diffusion**: conditional DDPM trained with noise prediction, deterministic DDIM sampling at inference

The thesis frames reconstruction as a problem of *prior selection*: with only a fraction of k-space measured, many images are consistent with the data, and each model resolves this ambiguity differently. All three models are trained *without* explicit data consistency, so that the effect of the learned prior can be examined on its own.

**Thesis:** *Accelerated MRI Reconstruction using CNNs, GANs and Diffusion Models* (Catholic University of Eichstätt-Ingolstadt, 2026).

## Setup

| | |
|---|---|
| Dataset | fastMRI brain, multi-coil, subset (`train_batch_0`, `train_batch_1`, `val_batch_0`) |
| Split | 901 train / 115 val / 115 test volumes (split at volume level, no slice leakage) |
| Test set | 1816 slices |
| Acceleration | R = 4, random Cartesian undersampling mask (fastMRI protocol) [TODO: confirm center fraction in `build_mask_r4`] |
| Input | zero-filled reconstruction, RSS-combined single-channel magnitude image |
| Preprocessing | official fastMRI transforms, center crop, per-image normalization, clipping to [−6, 6] |
| Training | batch size 1 per GPU (diffusion: 3 GPUs × 4 gradient accumulation steps = effective batch size 12), fixed seeds, two stages (epochs chosen on validation set, then retraining on train + val) |
| Hardware | NHR@FAU Alex cluster (A40/A100); CNN and GAN on 1 GPU, diffusion on 3 GPUs (DDP) |
| Software | PyTorch 2.9.1, CUDA 12.8 |

| | CNN | GAN | Diffusion |
|---|---|---|---|
| Epochs | 69 | 71 | 98 |
| Optimizer | Adam | Adam | AdamW |
| Learning rate | 3·10⁻⁴ | 2·10⁻⁴ | 3·10⁻⁵ |
| Model-specific | – | λ = 100 (L1 weight) | 400 timesteps, 200 DDIM steps, EMA, warmup, grad. clipping |

Hyperparameters were set per model for stable training and are **not** identical across models (see the thesis, Table 6.1).

## Results

Mean over the test set (full image, per-slice metrics, fastMRI protocol). The baseline is the zero-filled reconstruction.

| Model | NMSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| Zero-filled (baseline) | 0.0416 | 31.42 | 0.809 |
| CNN (U-Net) | 0.00882 | 37.97 | 0.939 |
| GAN (Pix2Pix) | 0.0126 | 36.46 | 0.910 |
| Diffusion (DDIM) | 0.2105 | 26.79 | 0.467 |

Restricted to the foreground (head) region:

| Model | NMSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| Zero-filled (baseline) | 0.0308 | 29.16 | 0.799 |
| CNN (U-Net) | 0.00750 | 34.77 | 0.918 |
| GAN (Pix2Pix) | 0.0103 | 33.36 | 0.884 |
| Diffusion (DDIM) | 0.0719 | 26.90 | 0.642 |

**Main observations**

- The CNN gives the best distortion metrics but produces over-smoothed reconstructions.
- The GAN gives sharper, more realistic-looking images at a higher risk of hallucinated structures.
- The diffusion model underperforms in this setting, including against the zero-filled baseline, with high variability and noise artifacts in the background.
- Beyond NMSE/PSNR/SSIM, the thesis analyses per-slice distributions, region-of-interest comparisons, edge-based structural errors (false and missed edges) and frequency-domain behavior. This repository contains the code for the metrics, foreground evaluation and visual/ROI analysis; [TODO: the scripts for the edge-based and frequency-domain analysis are not included yet].

## Limitations

Please read the results with these in mind:

- **The diffusion result should not be read as a statement about diffusion models in general.** The model was trained under constrained conditions (subset of fastMRI, batch size 1, limited training) and has not been tuned to the same degree as the CNN. Its failure to beat the zero-filled baseline suggests that training or the image-domain setup, rather than the method itself, may be a limiting factor. I have not isolated the cause.
- No explicit data consistency is used, which is the main reason for structural errors in all three models. Competitive reconstruction pipelines usually include it.
- Slices are reconstructed independently (2D, no volume context).
- Single run per model, single acceleration factor (R = 4), subset of the dataset.
- Hallucination risk is assessed qualitatively and through structural-error analysis, not through a dedicated metric.

## About this repository

This repository documents the code I used for the models and experiments in the thesis. It is **not** packaged as a ready-to-run tool: the scripts were written and run on the NHR@FAU cluster, so they depend on cluster-specific paths, shared helper modules and hardware settings (for example, the diffusion model trains on 3 GPUs with DDP), and some internal names (such as `model_3`) come from that environment. Datasets, checkpoints and training logs are not included.

The fastMRI data must be requested separately from the [fastMRI website](https://fastmri.med.nyu.edu/). The U-Net used for the CNN baseline and the data transforms come from the official [fastMRI repository](https://github.com/facebookresearch/fastMRI). Training uses PyTorch Lightning and TensorBoard.

## Repository structure

```
configs/
  cnn.yaml, gan.yaml, diffusion.yaml    # training configurations
scripts/
  cnn/         train_full_ds.py         # U-Net training
  gan/         train_full_ds.py         # Pix2Pix training
               generator.py, discriminator.py, gan_module.py, losses.py
  diffusion/   train_full_ds.py         # conditional DDPM training
               denoiser.py, diffusion_module.py, scheduler.py, sampling.py, losses.py
evaluation/                             # [TODO: currently named tests/; rename, these are evaluation scripts, not unit tests]
  metrics/     run_metrics_evaluation.py              # NMSE, PSNR, SSIM per slice (full image)
               run_metrics_evaluation_foreground.py   # same, restricted to the foreground region
               compute_metrics_statistics.py          # aggregate statistics
  visual/      select_visual_cases.py                 # choose representative slices
               generate_visual_cases.py               # reconstructions and error maps
               zoom_analysis.py                       # ROI comparison
```

Training hyperparameters are set at the top of each `train_full_ds.py`. [TODO: say how `configs/*.yaml` relates to them, or remove the folder if it is unused.]

## References

- J. Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*, 2018.
- O. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- P. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017.
- J. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- J. Song et al., *Denoising Diffusion Implicit Models*, 2020.
