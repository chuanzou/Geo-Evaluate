from __future__ import annotations

import torch


def multi_temporal_composite(
    cloudy_stack: torch.Tensor,
    mask_stack: torch.Tensor,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse temporal observations by averaging cloud-free pixels.

    Args:
        cloudy_stack: Tensor with shape TxCxHxW.
        mask_stack: Tensor with shape Tx1xHxW where 1 means cloud.
        fallback: Optional CxHxW tensor used where all observations are cloudy.
    """
    if cloudy_stack.ndim != 4 or mask_stack.ndim != 4:
        raise ValueError("cloudy_stack and mask_stack must be TxCxHxW and Tx1xHxW")

    clear = 1.0 - mask_stack
    weighted_sum = (cloudy_stack * clear).sum(dim=0)
    clear_count = clear.sum(dim=0).clamp_min(1e-6)
    composite = weighted_sum / clear_count

    if fallback is not None:
        no_clear = (clear.sum(dim=0) <= 1e-6).expand_as(composite)
        composite = torch.where(no_clear, fallback, composite)
    return composite.clamp(0.0, 1.0)
