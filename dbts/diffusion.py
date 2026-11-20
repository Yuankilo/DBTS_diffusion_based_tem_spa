"""Forward/reverse diffusion, time-frequency reconstruction loss, DCL joint loss, and scenario guidance."""

import math
import torch
from torch import nn

from .scenario import ScenarioGuidance


class Diffusion(nn.Module):
    """Conditional denoising diffusion wrapper implementing Eqs. (2)-(6) and (20)-(26)."""

    def __init__(
        self,
        model,
        steps=1000,
        lambda_time=1.0,
        lambda_fft=0.1,
        lambda_dcl=0.1,
        weight_mode="paper",
        lambda_s=1.0,
        objective="paper_l2",
        scenario_loss_weight=0.0,
        scenario_width=32,
    ):
        super().__init__()
        self.model = model
        self.steps = steps
        if steps < 2 or min(lambda_time, lambda_fft, lambda_dcl, lambda_s) < 0:
            raise ValueError("Invalid diffusion hyperparameters.")
        if lambda_dcl > 0 and not getattr(model, "use_dcl", False):
            raise ValueError("lambda_dcl > 0 requires a model with DCL enabled.")
        if objective not in ("paper_l2", "l1"):
            raise ValueError("objective must be 'paper_l2' or 'l1'.")
        if not math.isfinite(scenario_loss_weight) or scenario_loss_weight < 0:
            raise ValueError("scenario_loss_weight must be finite and nonnegative.")

        self.lambda_time = lambda_time
        self.lambda_fft = lambda_fft
        self.lambda_dcl = lambda_dcl
        self.objective = objective
        self.scenario_loss_weight = scenario_loss_weight
        self.scenario = (
            ScenarioGuidance(model.nodes, getattr(model, "dynamic_features", 0), scenario_width)
            if scenario_loss_weight
            else None
        )

        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        curve = curve / curve[0]
        beta = (1.0 - curve[1:] / curve[:-1]).clamp(max=0.999)
        alpha = 1.0 - beta
        ab = alpha.cumprod(0)
        prev = torch.cat([torch.ones(1, dtype=ab.dtype), ab[:-1]])

        if weight_mode == "paper":
            weights = lambda_s * (1.0 - alpha) / beta.square()
        elif weight_mode == "cosine":
            weights = alpha.sqrt() * (1.0 - ab).sqrt() / beta / 100.0
        elif weight_mode == "uniform":
            weights = torch.ones_like(beta)
        else:
            raise ValueError("weight_mode must be 'paper', 'cosine', or 'uniform'.")

        buffers = {
            "beta": beta,
            "alpha": alpha,
            "ab": ab,
            "prev": prev,
            "weights": weights,
            "posterior_var": beta * (1.0 - prev) / (1.0 - ab),
            "coef0": beta * prev.sqrt() / (1.0 - ab),
            "coeft": alpha.sqrt() * (1.0 - prev) / (1.0 - ab),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float())

    def q(self, x, t, noise):
        """Forward diffusion reparameterization (Eq. 2)."""
        a = self.ab[t].view(-1, 1, 1)
        return a.sqrt() * x + (1.0 - a).sqrt() * noise

    def loss(self, clean, dynamic=None, static=None, t=None, noise=None, origins=None):
        """Joint training objective combining time-frequency reconstruction (Eq. 25) and DCL (Eq. 26)."""
        t = torch.randint(self.steps, (len(clean),), device=clean.device) if t is None else t
        noise = torch.randn_like(clean) if noise is None else noise
        pred = self.model(self.q(clean, t, noise), t, dynamic, static)
        delta = pred - clean
        spectrum = torch.fft.fft(pred, dim=1, norm="forward") - torch.fft.fft(clean, dim=1, norm="forward")
        if self.objective == "paper_l2":
            time_loss = delta.square().mean((1, 2))
            fft_loss = spectrum.abs().square().mean((1, 2))
        else:
            time_loss = delta.abs().mean((1, 2))
            fft_loss = (spectrum.real.abs() + spectrum.imag.abs()).mean((1, 2))
        rec = (self.weights[t] * (self.lambda_time * time_loss + self.lambda_fft * fft_loss)).mean()
        dcl = self.model.dcl_loss(clean, dynamic, static, origins) if self.lambda_dcl else clean.new_zeros(())
        total = rec + self.lambda_dcl * dcl
        terms = {
            "time": time_loss.mean().detach(),
            "fft": fft_loss.mean().detach(),
            "weighted_reconstruction": rec.detach(),
            "dcl": dcl.detach(),
        }
        if self.scenario is not None:
            scenario_loss = self.scenario.training_loss(clean, dynamic)
            total = total + self.scenario_loss_weight * scenario_loss
            terms["scenario"] = scenario_loss.detach()
        terms["total"] = total.detach()
        return total, terms

    def guide_scenario(self, x, t, dynamic, static, reference, reference_strength, score_strength):
        """Dual-term condition-consistency gradient update (Eq. 6)."""
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            prediction = self.model(z, t, dynamic, static)
            objective = 0
            if reference_strength:
                distance = (prediction - reference).square().sum(dim=(1, 2))
                objective = -reference_strength * distance.sum()
            if score_strength:
                objective = objective + score_strength * self.scenario.log_probability(prediction, dynamic).sum()
            update = torch.autograd.grad(objective, z)[0]
        if not torch.isfinite(update).all():
            raise ValueError("Nonfinite scenario guidance update.")
        return (x + update).detach(), float(update.norm())

    def guide(self, x, t, history, dynamic, static, strength, score=None, score_strength=0.0):
        """Gradient guidance using observed-history discrepancy and an optional external score."""
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            pred = self.model(z, t, dynamic, static)
            error = (pred[:, : history.shape[1]] - history).square().mean((1, 2)).sum()
            objective = -strength * error
            if score_strength:
                val = score(pred, dynamic, static)
                if val.shape != (len(x),) or not torch.isfinite(val).all():
                    raise ValueError("Score must return finite [B] log-likelihood values.")
                objective = objective + score_strength * val.sum()
            grad = torch.autograd.grad(objective, z)[0]
        if not torch.isfinite(grad).all():
            raise ValueError("Nonfinite guidance gradient.")
        return (x + grad).detach(), float(grad.norm()), float(error.detach())

    @torch.no_grad()
    def forecast(
        self,
        history,
        horizon,
        *,
        dynamic=None,
        static=None,
        method="ddim",
        sampling_steps=50,
        seed=0,
        history_strength=0.0,
        score=None,
        score_strength=0.0,
        return_trace=False,
        schedule="endpoints",
        clip_x0=False,
        guidance_mode="scenario",
        reference_strength=0.0,
    ):
        """Generate conditional multi-step metro ridership forecasts via DDIM or DDPM reverse sampling."""
        if history.ndim != 3 or history.shape[1] + horizon != self.model.length or history.shape[2] != self.model.nodes:
            raise ValueError("history length + horizon must equal model sequence length.")
        if not torch.isfinite(history).all() or horizon < 1 or history.shape[1] < 1:
            raise ValueError("Invalid historical ridership input.")
        if method not in ("ddim", "ddpm") or not 1 <= sampling_steps <= self.steps:
            raise ValueError("Invalid sampling method or step count.")
        if method == "ddpm" and sampling_steps != self.steps:
            raise ValueError("DDPM requires full adjacent timesteps.")
        if guidance_mode not in ("scenario", "history_gradient"):
            raise ValueError("guidance_mode must be 'scenario' or 'history_gradient'.")

        scenario_mode = guidance_mode == "scenario"
        if scenario_mode and (reference_strength or score_strength):
            if self.scenario is None or dynamic is None:
                raise ValueError("Scenario guidance requires ScenarioGuidance module and dynamic inputs.")
        if not scenario_mode and score_strength and score is None:
            raise ValueError("External score function required when score_strength > 0.")

        if schedule == "uniform":
            timesteps = torch.linspace(-1, self.steps - 1, sampling_steps + 1).long().flip(0).tolist()[:-1]
        elif schedule == "endpoints":
            timesteps = [self.steps - 1] if (method == "ddim" and sampling_steps == 1) else torch.linspace(self.steps - 1, 0, sampling_steps).round().long().tolist()
        else:
            raise ValueError("schedule must be 'endpoints' or 'uniform'.")

        gen = torch.Generator(device=history.device).manual_seed(seed)
        rand = lambda shape: torch.randn(shape, device=history.device, dtype=history.dtype, generator=gen)
        b, h, n = history.shape
        x = rand((b, h + horizon, n))
        obsnoise = rand(history.shape)
        scenario_active = scenario_mode and bool(reference_strength or score_strength)
        reference = self.scenario.reference(dynamic) if (scenario_active and reference_strength) else None

        trace = []
        was_training = self.model.training
        self.model.eval()
        try:
            for i, s in enumerate(timesteps):
                nxt = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t = torch.full((b,), s, device=x.device, dtype=torch.long)
                x[:, :h] = self.q(history, t, obsnoise)
                gn = 0.0
                err = None

                if scenario_active:
                    x, gn = self.guide_scenario(x, t, dynamic, static, reference, reference_strength, score_strength)
                    x[:, :h] = self.q(history, t, obsnoise)
                elif not scenario_mode and (history_strength or score_strength):
                    x, gn, err = self.guide(x, t, history, dynamic, static, history_strength, score, score_strength)

                pred = self.model(x, t, dynamic, static)
                if clip_x0:
                    pred = pred.clamp(-1.0, 1.0)
                eps = (x - self.ab[s].sqrt() * pred) / (1.0 - self.ab[s]).sqrt()
                if method == "ddpm":
                    x = self.coef0[s] * pred + self.coeft[s] * x
                    if s > 0:
                        x = x + self.posterior_var[s].sqrt() * rand(x.shape)
                else:
                    an = self.ab[nxt] if nxt >= 0 else x.new_tensor(1.0)
                    x = an.sqrt() * pred + (1.0 - an).sqrt() * eps

                if nxt >= 0:
                    x[:, :h] = self.q(history, torch.full_like(t, nxt), obsnoise)
                else:
                    x[:, :h] = history
                trace.append({"t": s, "next": nxt, "gradient_norm": gn, "history_error": err})
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite forecast encountered.")
        finally:
            self.model.train(was_training)
        return (x, trace) if return_trace else x[:, h:]
