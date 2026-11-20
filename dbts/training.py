"""Training loop, EMA maintenance, learning-rate scheduling, and checkpointing."""

import copy
import random
import numpy as np
import torch

from .layers.lr_sch import ReduceLROnPlateauWithWarmup


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, lr=1e-4, ema_decay=0.995, scheduler=None, ema_mode="constant", ema_update_every=10):
        self.model = model
        self.ema = copy.deepcopy(model).eval()
        self.ema.requires_grad_(False)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.96))
        self.scheduler_config = scheduler
        if scheduler:
            self.scheduler = ReduceLROnPlateauWithWarmup(self.opt, **scheduler)
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda _: 1.0)
        self.ema_helper = None
        if ema_mode == "ema_pytorch":
            from ema_pytorch import EMA

            self.ema_helper = EMA(
                self.model,
                ema_model=self.ema,
                beta=ema_decay,
                update_every=ema_update_every,
                include_online_model=False,
            )
        elif ema_mode != "constant":
            raise ValueError("ema_mode must be 'constant' or 'ema_pytorch'.")
        self.step = 0
        self.ema_decay = ema_decay

    def update(self, x, **kwargs):
        return self.update_many([(x, kwargs)])

    def update_many(self, batches):
        if not batches:
            raise ValueError("Empty gradient accumulation batch list.")
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        terms = {}
        count = len(batches)
        for x, kwargs in batches:
            loss, part = self.model.loss(x, **kwargs)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss encountered.")
            (loss / count).backward()
            for k, v in part.items():
                terms[k] = terms.get(k, 0.0) + float(v) / count
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        lr_used = self.opt.param_groups[0]["lr"]
        self.opt.step()
        if self.scheduler_config:
            self.scheduler.step(terms["total"])
        else:
            self.scheduler.step()
        self.step += 1
        with torch.no_grad():
            if self.ema_helper is not None:
                self.ema_helper.update()
            else:
                for ep, p in zip(self.ema.parameters(), self.model.parameters()):
                    ep.lerp_(p, 1.0 - self.ema_decay)
                for eb, b in zip(self.ema.buffers(), self.model.buffers()):
                    eb.copy_(b)
        return {
            **terms,
            "gradient_norm": float(grad),
            "step": self.step,
            "learning_rate": lr_used,
            "gradient_accumulation": count,
        }

    def save(self, path, metadata):
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "ema_decay": self.ema_decay,
                "metadata": metadata,
                "ema_helper": self.ema_helper.state_dict() if self.ema_helper is not None else None,
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )

    def load(self, path, expected_metadata=None):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if expected_metadata is not None and data["metadata"] != expected_metadata:
            raise ValueError("Checkpoint configuration mismatch.")
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.opt.load_state_dict(data["optimizer"])
        self.scheduler.load_state_dict(data["scheduler"])
        if self.scheduler_config:
            self.scheduler.__dict__.update(data["scheduler"])
        if self.ema_helper is not None and data.get("ema_helper") is not None:
            self.ema_helper.load_state_dict(data["ema_helper"])
        self.step = data["step"]
        self.ema_decay = data["ema_decay"]
        random.setstate(data["random"])
        np.random.set_state(data["numpy"])
        torch.set_rng_state(data["torch"])
        if data["cuda"] is not None:
            torch.cuda.set_rng_state_all(data["cuda"])
        return data["metadata"]
