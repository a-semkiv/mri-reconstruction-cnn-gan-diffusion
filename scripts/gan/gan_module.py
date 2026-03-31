import pytorch_lightning as pl
import torch

from fastmri.pl_modules import UnetModule
from fastmri.evaluate import nmse, psnr, ssim

from scripts.model_2.discriminator import PatchDiscriminator
from scripts.model_2.losses import gan_loss, l1_loss


class Pix2PixModule(pl.LightningModule):

    def __init__(self, lr: float = 2e-4, lambda_l1: float = 100.0):
        super().__init__()

        self.generator = UnetModule(
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0,
            lr=0.0,  
        )

        self.discriminator = PatchDiscriminator(in_chans=2)

        self.lr = lr
        self.lambda_l1 = lambda_l1

        self.automatic_optimization = False


    def forward(self, masked: torch.Tensor) -> torch.Tensor:

        return self.generator(masked)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        masked, target, *_ = batch

        with torch.no_grad():
            fake = self.generator(masked)  

        
        masked_d = masked.unsqueeze(1)   
        target_d = target.unsqueeze(1)   
        fake_d   = fake.unsqueeze(1)     

        real_pair = torch.cat([masked_d, target_d], dim=1)  
        fake_pair = torch.cat([masked_d, fake_d], dim=1)    

        pred_real = self.discriminator(real_pair)
        pred_fake = self.discriminator(fake_pair)

        d_loss = 0.5 * (
            gan_loss(pred_real, is_real=True) +
            gan_loss(pred_fake, is_real=False)
        )

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()


        fake = self.generator(masked)  
        fake_d = fake.unsqueeze(1)     

        fake_pair = torch.cat([masked_d, fake_d], dim=1)
        pred_fake = self.discriminator(fake_pair)

        g_adv = gan_loss(pred_fake, is_real=True)
        g_l1 = l1_loss(fake, target)

        g_loss = g_adv + self.lambda_l1 * g_l1

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()


        self.log_dict(
            {
                "train/g_loss": g_loss,
                "train/d_loss": d_loss,
                "train/l1": g_l1,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )


    def validation_step(self, batch, batch_idx):
        masked, target, *_ = batch

        recon = self.generator(masked)  

        val_l1 = l1_loss(recon, target)

        target_np = target.detach().cpu().numpy()
        recon_np  = recon.detach().cpu().numpy()

        nmse_val = float(nmse(target_np, recon_np))
        psnr_val = float(psnr(target_np, recon_np))
        ssim_val = float(ssim(target_np, recon_np))

        self.log_dict(
            {
                "val/l1": val_l1,
                "val/nmse": nmse_val,
                "val/psnr": psnr_val,
                "val/ssim": ssim_val,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
        )

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
        )

        return [opt_g, opt_d]