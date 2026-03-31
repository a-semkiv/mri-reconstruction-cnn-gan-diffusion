# MRI Reconstruction: CNN vs GAN vs Diffusion

Implementation for a bachelor thesis on accelerated MRI reconstruction using three approaches:  
CNN (U-Net), GAN (Pix2Pix), and diffusion (DDPM) models.

## Dataset

Experiments are based on the fastMRI dataset (not included).  
(https://fastmri.med.nyu.edu/)

## Structure

configs/ # model configurations
scripts/ # training and model code
tests/ # evaluation and metrics

## Usage

Example (CNN): python scripts/cnn/train_full_ds.py

## Notes

- Training was performed on a compute cluster  
- Some paths may require adjustment  
- Checkpoints and full results are not included  
