import math
import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):

    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps

        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)

        alpha_bar = torch.cos(
            ((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2

        alpha_bar = alpha_bar / alpha_bar[0]

        alpha_bar_prev = alpha_bar[:-1]
        alpha_bar_curr = alpha_bar[1:]

        alphas = alpha_bar_curr / alpha_bar_prev
        betas = 1.0 - alphas
        betas = torch.clamp(betas, 1e-5, 0.999)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

        self.register_buffer("alpha_bars", alpha_bar)

        self.register_buffer(
            "sqrt_alpha_bars",
            torch.sqrt(alpha_bar)
        )

        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bar)
        )


    def sample_timesteps(self, batch_size: int):

        return torch.randint(
            1,
            self.num_timesteps + 1,
            (batch_size,),
            device=self.betas.device,
        )

    def q_sample(self, x_start, t, noise):

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        return sqrt_ab * x_start + sqrt_omb * noise


    def predict_start_from_noise(self, x_t, t, noise):

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        return (x_t - sqrt_omb * noise) / sqrt_ab