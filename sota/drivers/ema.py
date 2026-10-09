#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ema.py -- R30 passive parameter-EMA shadow.

Definition (exactly as specified, no warmup, no bias correction, fixed beta):

    theta_ema_0 = theta_online_0
    theta_ema_t = beta * theta_ema_(t-1) + (1-beta) * theta_online_t

where ``theta_online_t`` is the parameter value AFTER the t-th ``optimizer.step()``.

Scope
-----
Only parameters whose ONLINE ``requires_grad`` is True are averaged (the fusion
trunk + the five trainable delta heads + their trainable GroupNorm affines).
The two frozen towers' parameters and every buffer (including BatchNorm
``running_mean`` / ``running_var`` / ``num_batches_tracked``) are NOT averaged --
they are copied exactly from the online model.

The shadow is a deepcopy of the online model taken once at construction, so no
network is re-instantiated and no RNG is consumed.
"""
from __future__ import annotations

import copy

import torch


class ParameterEMA:
    """Passive EMA shadow of the trainable subset of an online model."""

    def __init__(self, online_model, beta: float = 0.999, whitelist=None):
        self.beta = float(beta)

        # Capture the whitelist from the ONLINE model *before* freezing the
        # shadow, so `shadow.requires_grad_(False)` cannot lose any name.
        if whitelist is None:
            whitelist = [n for n, p in online_model.named_parameters() if p.requires_grad]
        self.whitelist = list(whitelist)

        # deepcopy: no re-instantiation, no RNG consumption.
        self.shadow = copy.deepcopy(online_model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

        self._shadow_params = dict(self.shadow.named_parameters())
        self._online_params = dict(online_model.named_parameters())
        self._shadow_buffers = dict(self.shadow.named_buffers())
        self._online_buffers = dict(online_model.named_buffers())

        missing = [n for n in self.whitelist
                   if n not in self._shadow_params or n not in self._online_params]
        if missing:
            raise RuntimeError(f"EMA whitelist names absent: {missing[:5]}")

        wl = set(self.whitelist)
        self.n_ema_tensors = len(self.whitelist)
        self.n_ema_params = int(sum(self._shadow_params[n].numel() for n in self.whitelist))
        self.n_frozen_params = int(sum(p.numel() for n, p in self._online_params.items()
                                       if n not in wl))
        self.n_buffers = len(self._shadow_buffers)
        self.n_updates = 0

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self) -> None:
        """One EMA step. Pure arithmetic on the shadow -- no RNG, no autograd."""
        b = self.beta
        a = 1.0 - b
        for n in self.whitelist:
            self._shadow_params[n].mul_(b).add_(self._online_params[n].detach(), alpha=a)
        self.n_updates += 1

    # ------------------------------------------------- copy the non-EMA state
    @torch.no_grad()
    def sync_non_ema_state(self) -> None:
        """Copy every buffer and every non-whitelisted param exactly (no averaging)."""
        for n, sb in self._shadow_buffers.items():
            sb.copy_(self._online_buffers[n])
        wl = set(self.whitelist)
        for n, sp in self._shadow_params.items():
            if n not in wl:
                sp.copy_(self._online_params[n])

    # ------------------------------------------------------------------ misc
    def info(self) -> dict:
        return {
            "beta": self.beta,
            "n_updates": self.n_updates,
            "n_ema_tensors": self.n_ema_tensors,
            "n_ema_params": self.n_ema_params,
            "n_frozen_params_copied": self.n_frozen_params,
            "n_buffers_copied": self.n_buffers,
            "whitelist": list(self.whitelist),
        }
