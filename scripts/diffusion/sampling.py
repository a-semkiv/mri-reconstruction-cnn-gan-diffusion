import torch

@torch.no_grad()
def ddim_sample(
    denoiser,
    scheduler,
    masked_image,
    num_steps: int = None,
    debug: bool = False,
    return_trajectory: bool = False,
):
    trajectory = []
    trajectory_ts = []
    device = masked_image.device
    B, _, H, W = masked_image.shape
    T = scheduler.num_timesteps

    if num_steps is None or num_steps >= T:
        timesteps = torch.arange(T - 1, -1, -1, device=device)
    else:
        timesteps = torch.linspace(
            T - 1, 0, num_steps, device=device
        ).long()

    x = torch.randn((B, 1, H, W), device=device)

    save_steps = set([
        0,
        len(timesteps)//4,
        len(timesteps)//2,
        3*len(timesteps)//4,
        len(timesteps)-1,
    ])

    for i in range(len(timesteps) - 1):

        t = timesteps[i]
        t_prev = timesteps[i + 1]

        t_batch = torch.full(
            (B,), t, device=device, dtype=torch.long
        )

        eps_hat = denoiser(
            x_t=x,
            masked=masked_image,
            t=t_batch,
        )

        alpha_bar_t = scheduler.alpha_bars[t].view(1, 1, 1, 1)
        alpha_bar_prev = scheduler.alpha_bars[t_prev].view(1, 1, 1, 1)


        x0_hat = (
            x - torch.sqrt(1.0 - alpha_bar_t) * eps_hat
        ) / torch.sqrt(alpha_bar_t + 1e-8)

        x0_hat = torch.clamp(x0_hat, -6.0, 6.0)
        
        x = (
            torch.sqrt(alpha_bar_prev) * x0_hat +
            torch.sqrt(1.0 - alpha_bar_prev) * eps_hat
        )

        if return_trajectory and i in save_steps:
            trajectory.append(x.detach().cpu())
            trajectory_ts.append(t)

        if debug:
            print(
                f"step {i} | x min/max:",
                x.min().item(),
                x.max().item(),
            )

    if return_trajectory:
        trajectory.append(x0_hat.detach().cpu())
        trajectory_ts.append(0)  
        return x0_hat, trajectory, trajectory_ts

    return x0_hat