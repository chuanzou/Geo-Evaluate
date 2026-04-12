from __future__ import annotations

import torch
from torch import nn

from cloud_removal.models.unet import UNetGenerator


class InpaintingDiffusion(nn.Module):
    """Compact DDPM-style model for mask-guided inpainting experiments."""

    def __init__(self, channels: int = 4, timesteps: int = 100):
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.denoiser = UNetGenerator(in_channels=channels + 2, out_channels=channels)

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def forward(self, cloudy: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = target.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=target.device)
        noise = torch.randn_like(target)
        noisy = self.add_noise(target, t, noise)
        t_channel = t.float().view(batch, 1, 1, 1) / max(self.timesteps - 1, 1)
        t_channel = t_channel.expand(-1, 1, target.shape[-2], target.shape[-1])
        conditioned = noisy * mask + cloudy * (1.0 - mask)
        predicted_noise = self.denoiser(torch.cat([conditioned, mask, t_channel], dim=1))
        return predicted_noise, noise

    @torch.no_grad()
    def inpaint(self, cloudy: torch.Tensor, mask: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.timesteps
        x = torch.randn_like(cloudy)
        x = x * mask + cloudy * (1.0 - mask)

        for step in reversed(range(0, self.timesteps, max(self.timesteps // steps, 1))):
            t_channel = torch.full(
                (cloudy.shape[0], 1, cloudy.shape[-2], cloudy.shape[-1]),
                step / max(self.timesteps - 1, 1),
                device=cloudy.device,
            )
            predicted_noise = self.denoiser(torch.cat([x, mask, t_channel], dim=1))
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            x = (x - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
            x = x * mask + cloudy * (1.0 - mask)
        return x.clamp(0.0, 1.0)
