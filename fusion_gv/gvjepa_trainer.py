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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion_gv.gvjepa import FusionGVJEPA
from fusion_gv.preprocess import preprocess


# ── Grounding helpers ──────────────────────────────────────────────────────────

def boxes_to_patch_mask(
    boxes: list[list[float] | None],
    patch_grid: int = 37,
    patch_size: int = 14,
) -> torch.Tensor:
    """Convert per-view 2D boxes (518×518 space) to binary patch masks.

    Args:
        boxes:      list of S entries, each [x1,y1,x2,y2] or None (not visible)
        patch_grid: G, default 37
        patch_size: pixels per patch, default 14

    Returns:
        masks: (S, G, G) float32 binary patch mask
    """
    masks = []
    for box in boxes:
        mask = torch.zeros(patch_grid, patch_grid, dtype=torch.float32)
        if box is not None:
            x1, y1, x2, y2 = box
            px1 = max(0, int(x1 / patch_size))
            py1 = max(0, int(y1 / patch_size))
            px2 = min(patch_grid - 1, int(x2 / patch_size))
            py2 = min(patch_grid - 1, int(y2 / patch_size))
            if px2 > px1 and py2 > py1:
                mask[py1:py2 + 1, px1:px2 + 1] = 1.0
        masks.append(mask)
    return torch.stack(masks)   # (S, G, G)


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

    def __init__(self, manifest_path: str | Path, num_frames: int | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.num_frames = num_frames
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        if num_frames is not None:
            before = len(self.samples)
            self.samples = [
                s for s in self.samples
                if len(s.get("images", [s.get("image")])) >= num_frames
            ]
            dropped = before - len(self.samples)
            if dropped:
                print(f"[GVJEPADataset] {self.manifest_path.name}: dropped {dropped}/{before} samples with <{num_frames} frames")

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

        if self.num_frames is not None:
            image_paths = image_paths[: self.num_frames]

        return {
            "image_paths": image_paths,
            "query": row.get("query", ""),
            "target": row["target"],
            "boxes": row.get("boxes", None),   # list of S × ([x1,y1,x2,y2] | None) or None
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
    queries, targets, boxes_list = [], [], []

    for sample in batch:
        imgs_v, imgs_j = preprocess(sample["image_paths"])
        # imgs_v : (1, S, 3, 518, 518)
        # imgs_j : (S,  3, 1, 384, 384)
        vggt_list.append(imgs_v)
        jepa_list.append(imgs_j)
        queries.append(sample["query"])
        targets.append(sample["target"])
        boxes_list.append(sample["boxes"])   # list of S boxes or None

    images_vggt = torch.cat(vggt_list, dim=0)   # (B, S, 3, 518, 518)
    images_jepa = torch.cat(
        [j.unsqueeze(0) for j in jepa_list], dim=0
    ).flatten(0, 1)                              # (B*S, 3, 1, 384, 384)

    return {
        "images_vggt": images_vggt,
        "images_jepa": images_jepa,
        "query": queries,
        "target": targets,
        "boxes": boxes_list,   # list of B × (list of S boxes | None)
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
        grounding_loader: Optional[DataLoader] = None,
        grounding_ratio: int = 5,
        grad_accum_steps: int = 1,
        clip_grad_norm: float = 1.0,
        temperature: float = 0.07,
        log_every: int = 20,
        save_every: int = 1000,
        grounding_loss_weight: float = 0.1,
        grounding_pos_weight: float = 5.0,
        suppression_loss_weight: float = 0.1,
        mlflow_logger=None,
        accelerator=None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("`max_steps` must be > 0.")

        from accelerate import Accelerator
        self.accelerator = accelerator or Accelerator()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.grounding_loader = grounding_loader
        self.grounding_ratio = grounding_ratio
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.temperature = temperature
        self.log_every = log_every
        self.save_every = save_every
        self.mlflow_logger = mlflow_logger
        self.grounding_loss_weight   = grounding_loss_weight
        self.grounding_pos_weight    = grounding_pos_weight
        self.suppression_loss_weight = suppression_loss_weight
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_ckpt(self, step: int, filename: str | None = None) -> Path:
        raw_model = self.accelerator.unwrap_model(self.model)
        ckpt = {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        path = self.output_dir / (filename or f"step_{step:07d}.pt")
        torch.save(ckpt, path)
        return path

    def _write_meta(self) -> Path:
        meta = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.accelerator.device),
            "num_processes": self.accelerator.num_processes,
            "hostname": platform.node(),
        }
        path = self.output_dir / "run_meta.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        return path

    def _grounding_step(self, batch: dict) -> tuple[torch.Tensor, float]:
        """Forward + grounding BCE only. Returns (loss, grounding_loss_val)."""
        device = self.accelerator.device
        images_vggt = batch["images_vggt"].to(device, non_blocking=True)
        images_jepa = batch["images_jepa"].to(device, non_blocking=True)

        raw_model = self.accelerator.unwrap_model(self.model)
        with self.accelerator.autocast():
            grounding_loss = torch.tensor(0.0, device=device)
            boxes_batch = batch.get("boxes")
            if (
                raw_model.grounding_head is not None
                and boxes_batch is not None
                and any(b is not None for b in boxes_batch)
            ):
                patch_grid = raw_model.config.grounding_patch_grid
                gt_masks = torch.stack([
                    boxes_to_patch_mask(b, patch_grid)
                    if b is not None
                    else torch.zeros(images_vggt.shape[1], patch_grid, patch_grid)
                    for b in boxes_batch
                ]).to(device)
                B_g, S_g = gt_masks.shape[:2]
                gt_flat = gt_masks.reshape(B_g * S_g, patch_grid, patch_grid)
                valid = gt_flat.reshape(B_g * S_g, -1).sum(dim=-1) > 0

                logits = self.model.forward_grounding(images_vggt, images_jepa, batch["query"])

                if valid.any():
                    pw = torch.tensor(self.grounding_pos_weight, device=device, dtype=logits.dtype)
                    grounding_loss = F.binary_cross_entropy_with_logits(
                        logits[valid], gt_flat[valid], pos_weight=pw
                    )

                invisible = ~valid
                if invisible.any() and self.suppression_loss_weight > 0:
                    suppression_loss = logits[invisible].pow(2).mean()
                    grounding_loss = grounding_loss + self.suppression_loss_weight * suppression_loss

            loss = self.grounding_loss_weight * grounding_loss / self.grad_accum_steps

        return loss, grounding_loss.item()

    def _spar_step(self, batch: dict) -> tuple[torch.Tensor, float]:
        """Forward + InfoNCE only. Returns (loss, infonce_loss_val)."""
        device = self.accelerator.device
        images_vggt = batch["images_vggt"].to(device, non_blocking=True)
        images_jepa = batch["images_jepa"].to(device, non_blocking=True)

        with self.accelerator.autocast():
            out = self.model(images_vggt, images_jepa, batch["query"], batch["target"])
            loss = bidirectional_infonce(out["pred"], out["target"], temperature=self.temperature)
            loss = loss / self.grad_accum_steps

        return loss, loss.item() * self.grad_accum_steps

    def fit(self) -> None:
        """Run training until max_steps is reached."""
        import itertools

        self.model.train()
        is_main = self.accelerator.is_main_process
        meta_path = self._write_meta() if is_main else None
        log_path = self.output_dir / "train_log.jsonl"

        spar_iter = itertools.cycle(self.train_loader)
        grounding_iter = itertools.cycle(self.grounding_loader) if self.grounding_loader else None

        step = 0
        running_loss = 0.0
        running_infonce = 0.0
        running_grounding = 0.0
        pbar = tqdm(total=self.max_steps, desc="gvjepa-train", disable=not is_main)

        while step < self.max_steps:
            use_grounding = (
                grounding_iter is not None
                and step % self.grounding_ratio == 0
            )

            if use_grounding:
                batch = next(grounding_iter)
                loss, grounding_val = self._grounding_step(batch)
                infonce_val = 0.0
            else:
                batch = next(spar_iter)
                loss, infonce_val = self._spar_step(batch)
                grounding_val = 0.0

            self.accelerator.backward(loss)
            running_loss      += loss.item() * self.grad_accum_steps
            running_infonce   += infonce_val
            running_grounding += grounding_val

            if (step + 1) % self.grad_accum_steps == 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            step += 1
            pbar.update(1)

            if step % self.log_every == 0:
                entry = {
                    "step": step,
                    "loss": running_loss / self.log_every,
                    "infonce_loss": running_infonce / self.log_every,
                    "grounding_loss": running_grounding / self.log_every,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                if is_main:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    if self.mlflow_logger is not None:
                        self.mlflow_logger.log_metrics(
                            {
                                "train/loss":           entry["loss"],
                                "train/infonce_loss":   entry["infonce_loss"],
                                "train/grounding_loss": entry["grounding_loss"],
                                "train/lr":             entry["lr"],
                            },
                            step=step,
                        )
                running_loss = running_infonce = running_grounding = 0.0

            if is_main and self.save_every and self.save_every > 0 and step % self.save_every == 0:
                self._save_ckpt(step)

            if step >= self.max_steps:
                break

        if is_main:
            final_ckpt = self._save_ckpt(step, filename="final.pt")
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_artifacts([final_ckpt, meta_path, log_path])
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
    g = cfg.get("grounding", {})
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
        grounding_enabled=g.get("enabled", False),
        grounding_num_layers=g.get("num_layers", 3),
        grounding_num_heads=g.get("num_heads", 8),
        grounding_ffn_mult=g.get("ffn_mult", 4),
        grounding_dropout=g.get("dropout", 0.0),
        grounding_patch_grid=g.get("patch_grid", 37),
    )
    return FusionGVJEPA(model_cfg)


def _build_loader(manifests: list[str], dcfg: dict, batch_size: int, num_frames: int | None = None) -> DataLoader:
    """Build a DataLoader from a list of manifest paths and data config."""
    from pathlib import Path
    from torch.utils.data import ConcatDataset

    datasets = []
    for path in manifests:
        if not Path(path).exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        datasets.append(GVJEPADataset(path, num_frames=num_frames))

    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    num_workers = dcfg.get("num_workers", 4)
    loader_kwargs = {
        "batch_size": batch_size,
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


def build_loader_from_config(cfg: dict) -> DataLoader:
    """Build SPAR DataLoader from config (uses data.spar_manifests or data.train_manifests)."""
    dcfg = cfg["data"]
    manifests = dcfg.get("spar_manifests") or dcfg["train_manifests"]
    return _build_loader(manifests, dcfg, cfg["train"]["batch_size"])


def build_grounding_loader_from_config(cfg: dict) -> Optional[DataLoader]:
    """Build EmbodiedScan grounding DataLoader from config. Returns None if not configured."""
    dcfg = cfg["data"]
    manifests = dcfg.get("grounding_manifests")
    if not manifests:
        return None
    num_frames = dcfg.get("num_frames", None)
    return _build_loader(manifests, dcfg, cfg["train"]["batch_size"], num_frames=num_frames)


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
