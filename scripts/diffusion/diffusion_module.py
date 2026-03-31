import copy
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from fastmri.evaluate import nmse, psnr, ssim


from scripts.model_3.denoiser import DenoiserUNet
from scripts.model_3.scheduler import DiffusionScheduler
from scripts.model_3.losses import diffusion_noise_mse
from scripts.model_3.sampling import ddim_sample


class DiffusionModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 3e-6,
        num_timesteps: int = 400,
        chans: int = 48,
        num_pool_layers: int = 4,
        sampling_steps: int = 200,
        val_sampling_steps: int = 200,
        ema_decay: float = 0.9995,
        warmup_steps: int = 8000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.denoiser = DenoiserUNet(
            base_chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=0.05,
        )

        self.ema_decay = ema_decay
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.ema_denoiser.requires_grad_(False)

        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps
        )

        self.lr = lr
        self.sampling_steps = sampling_steps
        self.val_sampling_steps = val_sampling_steps
        self.warmup_steps = warmup_steps

    def forward(self, x_t, masked, t):
        return self.denoiser(x_t=x_t, masked=masked, t=t)


    def _ensure_4d(self, x):
        if x.dim() == 3:
            return x.unsqueeze(1)
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


    @torch.no_grad()
    def _update_ema(self):
        for ema_param, param in zip(
            self.ema_denoiser.parameters(),
            self.denoiser.parameters(),
        ):
            ema_param.data.mul_(self.ema_decay)
            ema_param.data.add_((1.0 - self.ema_decay) * param.data)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step > 1000:
            self._update_ema()


    def training_step(self, batch, batch_idx):

        masked, target, *_ = batch

        masked = self._ensure_4d(masked)
        target = self._ensure_4d(target)

        B = target.shape[0]

        x0 = target
        t = self.scheduler.sample_timesteps(B)
        alpha_bar = self.scheduler.alpha_bars[t] 

        p2_gamma = 0.5
        p2_k = 1.0

        weights = (p2_k + alpha_bar / (1 - alpha_bar)) ** (-p2_gamma)
        weights = weights.detach() 
        #

        noise = torch.randn_like(x0)

        x_t = self.scheduler.q_sample(x0, t, noise)

        pred_noise = self.denoiser(
            x_t=x_t,
            masked=masked,
            t=t,
        )

        mse = (pred_noise - noise) ** 2
        mse = mse.mean(dim=(1, 2, 3), keepdim=True)

        weighted_loss = (weights * mse).mean()

        self.log(
            "train/noise_loss",
            weighted_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )

        return weighted_loss


    def validation_step(self, batch, batch_idx):

        masked, target, mean, std, *_ = batch

        masked = self._ensure_4d(masked).to(self.device)
        target = self._ensure_4d(target).to(self.device)

        mean = mean.to(self.device)
        std = std.to(self.device)

        B = target.shape[0]

        x0 = target

        t = self.scheduler.sample_timesteps(B)

        noise = torch.randn_like(x0)

        x_t = self.scheduler.q_sample(x0, t, noise)

        pred_noise = self.denoiser(
            x_t=x_t,
            masked=masked,
            t=t,
        )

        val_noise_loss = F.mse_loss(pred_noise, noise)


        recon = self.sample(
            masked,
            num_steps=self.val_sampling_steps,
            use_ema=True,
        )

        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)

        recon = recon * std + mean
        target = target * std + mean

        val_l1 = F.l1_loss(recon, target)

        recon_np = recon.squeeze(1).detach().cpu().numpy()
        target_np = target.squeeze(1).detach().cpu().numpy()

        val_nmse = float(nmse(target_np, recon_np))
        val_psnr = float(psnr(target_np, recon_np))
        val_ssim = float(ssim(target_np, recon_np))

        self.log_dict(
            {
                "val/noise_loss": val_noise_loss,
                "val/l1": val_l1,
                "val/nmse": val_nmse,
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )

    @torch.no_grad()
    def sample(self, masked, num_steps=None, use_ema=False):

        if num_steps is None:
            num_steps = self.sampling_steps

        model = self.ema_denoiser if use_ema else self.denoiser

        return ddim_sample(
            denoiser=model,
            scheduler=self.scheduler,
            masked_image=masked,
            num_steps=num_steps,
            debug=False,
        )


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }