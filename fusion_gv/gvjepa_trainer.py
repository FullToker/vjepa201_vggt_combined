"""
Training utilities for FusionGVJEPA.

Dataset format (JSONL, one sample per line):
    {
      "images": ["path/to/frame1.jpg", "path/to/frame2.jpg", ...],
      "query":  "Describe the 3D structure of the scene.",
      "target": "A table with two chairs and a lamp on top."
    }

    Single-image variant (S=1):
    {"image": "path/to/img.jpg", "query": "...", "target": "..."}

Usage
-----
    from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig
    from fusion_gv.gvjepa_trainer import GVJEPADataset, gvjepa_collate, GVJEPATrainer
    from torch.utils.data import DataLoader

    cfg = GVJEPAConfig()
    model = FusionGVJEPA(cfg)

    dataset = GVJEPADataset("data/train.jsonl")
    loader  = DataLoader(dataset, batch_size=8, collate_fn=gvjepa_collate,
                         shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(
        model.parameter_groups(lr=1e-4, weight_decay=0.05)
    )
    trainer = GVJEPATrainer(model, optimizer, loader, output_dir="runs/gvjepa")
    trainer.fit(torch.device("cuda"))
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion_gv.gvjepa import FusionGVJEPA
from fusion_gv.preprocess import preprocess


# ── InfoNCE loss (bidirectional, paper Sec. 2) ─────────────────────────────────

def bidirectional_infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Bidirectional InfoNCE between (B, D) pred and (B, D) target tensors."""
    import torch.nn.functional as F

    if pred.shape[0] < 2:
        raise ValueError("InfoNCE requires batch size >= 2.")
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (pred @ target.T) / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# ── Dataset ────────────────────────────────────────────────────────────────────

class GVJEPADataset(Dataset):
    """JSONL dataset for FusionGVJEPA.

    Each row must contain:
        "images"  : list of image paths  (S ≥ 1)
            OR
        "image"   : single image path    (treated as S=1)
        "query"   : text query string
        "target"  : text target string
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        if "target" not in row:
            raise KeyError(f"Row {idx} missing required 'target' field.")

        if "images" in row:
            image_paths = row["images"]
        elif "image" in row:
            image_paths = [row["image"]]
        else:
            raise ValueError(f"Row {idx} must have 'images' or 'image'.")

        return {
            "image_paths": image_paths,
            "query": row.get("query", ""),
            "target": row["target"],
        }


def gvjepa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples into a batched dict with preprocessed tensors.

    Calls fusion_gv.preprocess() per sample so each sample can have a different
    number of images.  Batch requires all samples to have the same S (number of
    frames); pad or truncate in the dataset if needed.

    Returns:
        images_vggt : (B, S, 3, 518, 518)
        images_jepa : (B*S, 3, 1, 384, 384)
        queries     : list of B strings
        targets     : list of B strings
    """
    vggt_list, jepa_list = [], []
    queries, targets = [], []

    for sample in batch:
        imgs_v, imgs_j = preprocess(sample["image_paths"])
        # imgs_v : (1, S, 3, 518, 518)
        # imgs_j : (S,  3, 1, 384, 384)
        vggt_list.append(imgs_v)   # keep leading batch dim
        jepa_list.append(imgs_j)
        queries.append(sample["query"])
        targets.append(sample["target"])

    images_vggt = torch.cat(vggt_list, dim=0)   # (B, S, 3, 518, 518)
    images_jepa = torch.cat(
        [j.unsqueeze(0) for j in jepa_list], dim=0
    ).flatten(0, 1)                              # (B*S, 3, 1, 384, 384)

    return {
        "images_vggt": images_vggt,
        "images_jepa": images_jepa,
        "query": queries,
        "target": targets,
    }


# ── Trainer ────────────────────────────────────────────────────────────────────

class GVJEPATrainer:
    """
    Training loop for FusionGVJEPA.

    Matches VL-JEPA's training structure (paper Sec. 3.2):
    - Mixed precision (bf16 default)
    - Gradient accumulation + clipping
    - Optional cosine LR scheduler
    - JSONL training log + periodic checkpoints
    """

    def __init__(
        self,
        model: FusionGVJEPA,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        output_dir: str | Path,
        max_steps: int,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_accum_steps: int = 1,
        clip_grad_norm: float = 1.0,
        temperature: float = 0.07,
        log_every: int = 20,
        save_every: int = 1000,
        precision: str = "bf16",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("`max_steps` must be > 0.")
        if precision not in {"bf16", "fp16"}:
            raise ValueError("`precision` must be 'bf16' or 'fp16'.")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.temperature = temperature
        self.log_every = log_every
        self.save_every = save_every
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        self.autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_ckpt(self, step: int) -> None:
        ckpt = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(ckpt, self.output_dir / f"step_{step:07d}.pt")

    def _write_meta(self, device: torch.device) -> None:
        meta = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "hostname": platform.node(),
        }
        with open(self.output_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def fit(self, device: torch.device) -> None:
        """Run training until max_steps is reached."""
        self.model.to(device)
        self.model.train()
        self._write_meta(device)
        log_path = self.output_dir / "train_log.jsonl"

        step = 0
        running_loss = 0.0
        pbar = tqdm(total=self.max_steps, desc="gvjepa-train")

        while step < self.max_steps:
            for batch in self.train_loader:
                images_vggt = batch["images_vggt"].to(device, non_blocking=True)
                images_jepa = batch["images_jepa"].to(device, non_blocking=True)
                queries = batch["query"]
                targets = batch["target"]

                with torch.autocast(device_type=device.type, dtype=self.autocast_dtype):
                    out = self.model(images_vggt, images_jepa, queries, targets)
                    loss = bidirectional_infonce(
                        out["pred"], out["target"], temperature=self.temperature
                    )
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                running_loss += loss.item() * self.grad_accum_steps

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

                step += 1
                pbar.update(1)

                if step % self.log_every == 0:
                    entry = {
                        "step": step,
                        "loss": running_loss / self.log_every,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    running_loss = 0.0

                if step % self.save_every == 0:
                    self._save_ckpt(step)

                if step >= self.max_steps:
                    break

        self._save_ckpt(step)
        pbar.close()


# ── Low-level factory helpers (manual use) ─────────────────────────────────────

def build_optimizer(
    model: FusionGVJEPA,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    adam_betas: tuple[float, float] = (0.9, 0.98),
    adam_eps: float = 1e-6,
) -> torch.optim.AdamW:
    """AdamW with separate slow LR for Y-Encoder (paper Sec. 3.2)."""
    return torch.optim.AdamW(
        model.parameter_groups(lr=lr, weight_decay=weight_decay),
        betas=adam_betas,
        eps=adam_eps,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine annealing scheduler used in VL-JEPA SFT stage."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=min_lr
    )


# ── Config-dict factory helpers (YAML-driven use) ──────────────────────────────

def build_model_from_config(cfg: dict) -> "FusionGVJEPA":
    """Instantiate FusionGVJEPA from a parsed YAML config dict."""
    from fusion_gv.config import FusionConfig
    from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig

    f = cfg["fusion"]
    fusion_cfg = FusionConfig(
        x_encoder_type=f.get("x_encoder_type", "fusion_gv"),
        x_encoder_output_dim=f.get("x_encoder_output_dim"),
        vggt_ckpt=f.get("vggt_ckpt", "./ckpts/vggt.pt"),
        jepa_ckpt=f["jepa_ckpt"],
        fusion_type=f.get("fusion_type", "concat"),
        num_levels=f.get("num_levels", 4),
        # encoder geometry (use dataclass defaults if not specified)
        **{k: f[k] for k in (
            "vggt_img_size", "vggt_patch_size", "vggt_embed_dim",
            "vggt_out_dim", "vggt_num_patches",
            "jepa_img_size", "jepa_patch_size", "jepa_embed_dim", "jepa_num_patches",
        ) if k in f},
    )

    m = cfg["model"]
    model_cfg = GVJEPAConfig(
        fusion=fusion_cfg,
        use_fusion_level=m.get("use_fusion_level", 3),
        predictor_hidden_size=m["predictor_hidden_size"],
        predictor_layers=m["predictor_layers"],
        predictor_heads=m["predictor_heads"],
        predictor_ffn_mult=m["predictor_ffn_mult"],
        predictor_dropout=m.get("predictor_dropout", 0.0),
        query_model_name=m.get("query_model_name", "toy"),
        max_query_tokens=m.get("max_query_tokens", 64),
        y_encoder_name=m.get("y_encoder_name", "toy"),
        max_target_tokens=m.get("max_target_tokens", 64),
        shared_embed_dim=m["shared_embed_dim"],
        y_encoder_lr_multiplier=m.get("y_encoder_lr_multiplier", 0.05),
        hf_cache_dir=m.get("hf_cache_dir", "./ckpts"),
    )
    return FusionGVJEPA(model_cfg)


def build_loader_from_config(cfg: dict) -> DataLoader:
    """Build DataLoader from a parsed YAML config dict."""
    from pathlib import Path

    dcfg = cfg["data"]
    manifests = dcfg["train_manifests"]
    datasets = []
    for path in manifests:
        if not Path(path).exists():
            raise FileNotFoundError(f"Training manifest not found: {path}")
        datasets.append(GVJEPADataset(path))

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset(datasets)

    num_workers = dcfg.get("num_workers", 4)
    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": dcfg.get("pin_memory", True),
        "collate_fn": gvjepa_collate,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = dcfg.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = dcfg.get("prefetch_factor", 4)

    return DataLoader(ds, **loader_kwargs)


def build_optimizer_and_scheduler_from_config(
    model: "FusionGVJEPA",
    cfg: dict,
) -> tuple:
    """Build (optimizer, scheduler | None) from a parsed YAML config dict."""
    tcfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            lr=tcfg["learning_rate"],
            weight_decay=tcfg.get("weight_decay", 0.01),
        ),
        betas=(tcfg.get("adam_beta1", 0.9), tcfg.get("adam_beta2", 0.95)),
        eps=tcfg.get("adam_eps", 1e-8),
    )
    sched_name = tcfg.get("scheduler", "constant")
    if sched_name == "cosine":
        scheduler = build_cosine_scheduler(
            optimizer,
            max_steps=tcfg["max_steps"],
            min_lr=tcfg.get("min_lr", 0.0),
        )
    elif sched_name == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name!r}")
    return optimizer, scheduler
