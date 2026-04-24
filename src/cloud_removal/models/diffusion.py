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
        # CRITICAL: final_activation=None keeps the denoiser output linear so
        # it can predict noise values in (-inf, +inf). The default sigmoid
        # used by the GAN generator clamps outputs to [0, 1], which makes it
        # mathematically impossible to predict a noise tensor sampled from
        # N(0, 1) — half the targets are negative. Using sigmoid here was the
        # root cause of diffusion completely failing to learn (val MSE stuck
        # at ~0.585, L1_cloud stuck at ~0.27 across all coverages).
        self.denoiser = UNetGenerator(
            in_channels=channels + 2,
            out_channels=channels,
            final_activation=None,
        )

        # Linear schedule. We tried cosine first to fix the alpha_bar_T ~ 0.37
        # signal-leak with T=100, but cosine pushes beta_max to 0.999, which
        # the current naive sampler can't handle (division by sqrt(alpha)
        # explodes by 32x at the first step). Sticking with linear for now —
        # once the sigmoid bug is verified fixed and the model actually
        # learns, we can revisit schedule + sampler robustness.
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def forward_at_t(
        self,
        cloudy: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Single forward pass with caller-controlled t and noise.

        Split out so training / validation can log per-sample t, and so
        validation can use a deterministic fixed-t grid instead of random
        t — which makes the reported val-MSE stable across epochs and
        enables per-timestep analysis (e.g. heatmap of val-MSE vs t).
        """
        batch = target.shape[0]
        noisy = self.add_noise(target, t, noise)
        t_channel = t.float().view(batch, 1, 1, 1) / max(self.timesteps - 1, 1)
        t_channel = t_channel.expand(-1, 1, target.shape[-2], target.shape[-1])
        conditioned = noisy * mask + cloudy * (1.0 - mask)
        predicted_noise = self.denoiser(torch.cat([conditioned, mask, t_channel], dim=1))
        return predicted_noise

    def forward(
        self, cloudy: torch.Tensor, mask: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard training forward: random t per sample, random noise."""
        batch = target.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=target.device)
        noise = torch.randn_like(target)
        predicted_noise = self.forward_at_t(cloudy, mask, target, t, noise)
        return predicted_noise, noise, t

    @torch.no_grad()
    def inpaint(self, cloudy: torch.Tensor, mask: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """Reverse-diffusion sampling, with mask compositing each step.

        Numerically stable formulation: at every step we
          1. predict noise ε with the denoiser,
          2. recover an x0 estimate ``x0_pred = (x_t - sqrt(1-α̅_t) ε) / sqrt(α̅_t)``,
          3. CLAMP ``x0_pred`` into the valid image range [0, 1] (this is the
             single most important stabiliser — without it small per-step
             prediction errors get amplified by the inverse-α̅ factor and the
             chain diverges within a few iterations into garbage),
          4. step ``x_{t-1}`` from the posterior ``q(x_{t-1} | x_t, x0_pred)``
             whose closed-form mean / variance are given by Ho et al. 2020,
          5. composite the result so that pixels outside the cloud mask are
             always equal to the original ``cloudy`` observation.

        The ``steps`` argument is ignored; sampling always uses the full
        ``self.timesteps`` reverse chain. Subsampling needs more careful
        handling of α̅_{t-1} indexing and was a source of bugs previously.

        -------------------------------------------------------------------
        Init strategy (the one and only thing that actually works here).
        -------------------------------------------------------------------
        With T=100 linear β-schedule, α̅_{T-1} ≈ 0.37, so at training time the
        denoiser *always* saw inputs of the form
              x = sqrt(α̅)·target + sqrt(1-α̅)·noise.
        If we init with pure ``torch.randn`` the first step is OOD in *std*
        (1.0 vs the training 0.79). If we init with ``cloudy`` as a target
        proxy — what the previous implementation did — we are OOD in *mean*
        by a huge amount: inside the mask ``cloudy≈1`` but the real targets
        in this dataset have mean ≈ 0.12. This bias *compounds* along the
        DDPM chain and produces outputs that are 45–76% white (verified
        empirically: L1_cloud grows from 0.40 at 5% coverage to 0.67 at 70%,
        exactly matching ``α·(1-target_mean)`` for α = white-bias fraction).
        See ``scripts/diagnose_no_torch.py`` for the numbers.

        We therefore use a **per-image, per-channel outside-mask mean** as
        the target proxy. Outside the mask the synthetic setup guarantees
        ``cloudy == target``, so this mean is an unbiased estimate of the
        target's pixel statistics for *that specific image*. The resulting
        init x has the right mean and variance distribution for the
        denoiser's training regime.
        """
        del steps  # subsampling unsupported in this stable implementation

        alpha_bar_last = self.alpha_bars[-1]
        # --- compute per-image, per-channel clean-region mean as target proxy ---
        # cloudy: (B, C, H, W); mask: (B, 1, H, W) with 1 = cloud.
        clean_weight = (1.0 - mask)                                       # (B,1,H,W)
        clean_denom = clean_weight.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)  # (B,1,1,1)
        clean_mean = (cloudy * clean_weight).sum(dim=(-2, -1), keepdim=True) / clean_denom  # (B,C,1,1)
        target_proxy = clean_mean.expand_as(cloudy)                       # (B,C,H,W)

        noise_init = torch.randn_like(cloudy)
        x = alpha_bar_last.sqrt() * target_proxy + (1.0 - alpha_bar_last).sqrt() * noise_init
        x = x * mask + cloudy * (1.0 - mask)

        for step in reversed(range(self.timesteps)):
            t_channel = torch.full(
                (cloudy.shape[0], 1, cloudy.shape[-2], cloudy.shape[-1]),
                step / max(self.timesteps - 1, 1),
                device=cloudy.device,
            )
            predicted_noise = self.denoiser(torch.cat([x, mask, t_channel], dim=1))

            alpha_bar = self.alpha_bars[step]
            # x0 prediction (Tweedie / DDPM eq. 15) + clamp.
            x0_pred = (x - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(0.0, 1.0)

            if step > 0:
                alpha_bar_prev = self.alpha_bars[step - 1]
                beta = self.betas[step]
                alpha = self.alphas[step]
                # Posterior mean q(x_{t-1} | x_t, x0)  (DDPM eq. 7)
                coef_x0 = alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar)
                coef_xt = alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                mean = coef_x0 * x0_pred + coef_xt * x
                # Posterior variance σ_t² = β_t (1-α̅_{t-1}) / (1-α̅_t)
                var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                noise = torch.randn_like(x)
                x = mean + var.clamp(min=0.0).sqrt() * noise
            else:
                # At t=0 the posterior is deterministic: x_{-1} = x0_pred.
                x = x0_pred

            # Re-composite: keep the known clean pixels outside the cloud.
            x = x * mask + cloudy * (1.0 - mask)

        return x.clamp(0.0, 1.0)
