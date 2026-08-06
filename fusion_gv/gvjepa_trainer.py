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

import gc
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

def boxes_to_gaussian_heatmap(
    boxes: list[list[float] | None],
    patch_grid: int = 37,
    patch_size: int = 14,
    min_sigma: float = 0.5,
) -> torch.Tensor:
    """Convert per-view 2D boxes (518×518 space) to unnormalized Gaussian heatmaps
    (CenterNet-style ground truth — "Objects as Points", Zhou et al. 2019).

    Peak value 1.0 sits on the box's center patch, decaying outward with sigma
    set from box extent (±3σ ≈ box half-width/height). The center is rounded to
    its nearest grid cell before the Gaussian is evaluated so exactly one pixel
    hits 1.0 exactly — modified_focal_loss's positive-point mask depends on that
    exact equality to find it. Frames with box=None (target not visible in this
    view) get an all-zero heatmap.

    Args:
        boxes:      list of S entries, each [x1,y1,x2,y2] or None (not visible)
        patch_grid: G, default 37
        patch_size: pixels per patch, default 14
        min_sigma:  floor on sigma (patch units) so tiny boxes don't collapse
                    to a near-zero-width Gaussian

    Returns:
        heatmaps: (S, G, G) float32, values in [0, 1], peak 1.0 at box center
    """
    G = patch_grid
    ys = torch.arange(G, dtype=torch.float32).unsqueeze(1)   # (G,1)
    xs = torch.arange(G, dtype=torch.float32).unsqueeze(0)   # (1,G)

    heatmaps = []
    for box in boxes:
        hm = torch.zeros(G, G, dtype=torch.float32)
        if box is not None:
            x1, y1, x2, y2 = box
            px1, py1 = x1 / patch_size, y1 / patch_size
            px2, py2 = x2 / patch_size, y2 / patch_size
            if px2 > px1 and py2 > py1:
                cx = min(max(round((px1 + px2) / 2.0), 0), G - 1)
                cy = min(max(round((py1 + py2) / 2.0), 0), G - 1)
                sigma_x = max((px2 - px1) / 6.0, min_sigma)
                sigma_y = max((py2 - py1) / 6.0, min_sigma)
                hm = torch.exp(-(
                    (xs - cx) ** 2 / (2 * sigma_x ** 2)
                    + (ys - cy) ** 2 / (2 * sigma_y ** 2)
                ))
        heatmaps.append(hm)
    return torch.stack(heatmaps)   # (S, G, G)


def modified_focal_loss(
    logits: torch.Tensor,       # (N, G, G) raw pre-sigmoid logits
    gt_heatmap: torch.Tensor,   # (N, G, G) unnormalized Gaussian target, peak=1.0
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """CenterNet-style modified focal loss (CornerNet, Law & Deng 2018 /
    "Objects as Points", Zhou et al. 2019).

    Positive points (gt_heatmap == 1.0, i.e. box centers) use
    -(1-p)^alpha * log(p). Every other point uses
    -(1-Y)^beta * p^alpha * log(1-p), so points near a center are penalized
    less than far-away background, and confidently-correct background points
    are penalized less than confidently-wrong ones (standard focal behavior).

    Frames with no visible target (gt_heatmap all-zero) fall entirely into the
    negative branch with full weight ((1-Y)^beta = 1 everywhere) — this alone
    reproduces what a separate "suppression loss" would do, so no extra term
    is needed for invisible views.

    Normalizes by the number of positive (center) points in the batch, clamped
    to >= 1 so an all-invisible batch doesn't divide by zero.
    """
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pos_mask = gt_heatmap.eq(1.0).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -torch.log(p) * (1 - p).pow(alpha) * pos_mask
    neg_loss = -torch.log(1 - p) * p.pow(alpha) * (1 - gt_heatmap).pow(beta) * neg_mask

    num_pos = pos_mask.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


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

        # Index-only pass: record byte offsets, never retain parsed rows.
        # Keeps per-process memory to O(N) ints instead of O(N) dicts, so
        # DataLoader workers (fork'd copies) don't each balloon to a full
        # copy of a multi-million-row parsed manifest.
        self._offsets: list[int] = []
        before = 0
        with open(self.manifest_path, "rb") as f:
            offset = f.tell()
            for raw in f:
                if raw.strip():
                    before += 1
                    keep = True
                    if num_frames is not None:
                        row = json.loads(raw)
                        n_imgs = len(row.get("images", [row.get("image")]))
                        keep = n_imgs >= num_frames
                    if keep:
                        self._offsets.append(offset)
                offset = f.tell()
        if not self._offsets:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        dropped = before - len(self._offsets)
        if dropped:
            print(f"[GVJEPADataset] {self.manifest_path.name}: dropped {dropped}/{before} samples with <{num_frames} frames")

        # Opened lazily, once per worker process, on first __getitem__ call —
        # never before DataLoader forks workers (a pre-fork fd would share
        # its seek position across processes and race).
        self._fh = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_row(self, idx: int) -> Dict[str, Any]:
        if self._fh is None:
            self._fh = open(self.manifest_path, "rb")
        self._fh.seek(self._offsets[idx])
        return json.loads(self._fh.readline())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_row(idx)
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
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        grounding_ema_decay: float = 0.98,
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
        self.grounding_loss_weight = grounding_loss_weight
        self.focal_alpha           = focal_alpha
        self.focal_beta            = focal_beta
        self.grounding_ema_decay   = grounding_ema_decay
        # EMA of raw grounding_loss, self-normalizes it before applying
        # grounding_loss_weight (see _grounding_step). None until the first
        # real (non-placeholder) grounding loss is observed -- initialized to
        # that first value directly rather than decayed in from 0, so the
        # early steps aren't biased toward an artificial near-zero EMA.
        self._grounding_ema: float | None = None
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

    def _grounding_step(self, batch: dict) -> tuple[torch.Tensor, float, float]:
        """Forward + grounding modified-focal-loss only.

        Returns (loss, raw_grounding_loss_val, ema_normalized_grounding_loss_val).
        `loss` (what actually gets backpropped) uses the EMA-normalized value;
        the raw value is returned separately purely for logging/diagnostics —
        it's what you want to watch to see if grounding is actually improving,
        since the normalized value is deliberately flattened to ~1.0 by design.
        """
        device = self.accelerator.device
        images_vggt = batch["images_vggt"].to(device, non_blocking=True)
        images_jepa = batch["images_jepa"].to(device, non_blocking=True)

        raw_model = self.accelerator.unwrap_model(self.model)
        with self.accelerator.autocast():
            grounding_loss = torch.tensor(0.0, device=device)
            normalized_grounding_loss = grounding_loss
            boxes_batch = batch.get("boxes")
            if (
                raw_model.grounding_head is not None
                and boxes_batch is not None
                and any(b is not None for b in boxes_batch)
            ):
                patch_grid = raw_model.config.grounding_patch_grid
                gt_heatmaps = torch.stack([
                    boxes_to_gaussian_heatmap(b, patch_grid)
                    if b is not None
                    else torch.zeros(images_vggt.shape[1], patch_grid, patch_grid)
                    for b in boxes_batch
                ]).to(device)
                B_g, S_g = gt_heatmaps.shape[:2]
                gt_flat = gt_heatmaps.reshape(B_g * S_g, patch_grid, patch_grid)

                out = self.model(images_vggt, images_jepa, batch["query"], mode="grounding")
                logits = out["grounding_logits"]

                # invisible-view rows (gt_flat all-zero) fall entirely into the
                # negative branch of the focal loss — no separate handling needed.
                grounding_loss = modified_focal_loss(
                    logits, gt_flat, alpha=self.focal_alpha, beta=self.focal_beta
                )

                # EMA-normalize so grounding_loss_weight means roughly the same
                # thing throughout training, instead of tracking whatever raw
                # scale the focal loss happens to be at (large early, shrinks
                # as the model converges). ema is a detached python float, not
                # a tensor, so it never enters the autograd graph -- see the
                # note in the earlier conversation on why that matters.
                raw_val = grounding_loss.item()
                if self._grounding_ema is None:
                    self._grounding_ema = raw_val
                else:
                    self._grounding_ema = (
                        self.grounding_ema_decay * self._grounding_ema
                        + (1 - self.grounding_ema_decay) * raw_val
                    )
                normalized_grounding_loss = grounding_loss / (self._grounding_ema + 1e-8)

            loss = self.grounding_loss_weight * normalized_grounding_loss / self.grad_accum_steps

        return loss, grounding_loss.item(), normalized_grounding_loss.item()

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

        def _infinite(loader):
            # NOTE: itertools.cycle caches every yielded batch to replay after
            # the first pass — since accelerator.prepare() already places
            # batches on GPU, that cache retains one GPU batch per step
            # forever. Re-iterate the loader instead (also reshuffles each
            # epoch since shuffle=True).
            while True:
                yield from loader

        self.model.train()
        is_main = self.accelerator.is_main_process
        meta_path = self._write_meta() if is_main else None
        log_path = self.output_dir / "train_log.jsonl"

        spar_iter = _infinite(self.train_loader)
        grounding_iter = _infinite(self.grounding_loader) if self.grounding_loader else None

        step = 0
        running_loss = 0.0
        running_infonce = 0.0
        running_grounding = 0.0
        running_grounding_ema = 0.0
        pbar = tqdm(total=self.max_steps, desc="gvjepa-train", disable=not is_main)

        while step < self.max_steps:
            use_grounding = (
                grounding_iter is not None
                and step % self.grounding_ratio == 0
            )

            if use_grounding:
                batch = next(grounding_iter)
                loss, grounding_val, grounding_ema_val = self._grounding_step(batch)
                infonce_val = 0.0
            else:
                batch = next(spar_iter)
                loss, infonce_val = self._spar_step(batch)
                grounding_val = 0.0
                grounding_ema_val = 0.0

            self.accelerator.backward(loss)
            running_loss           += loss.item() * self.grad_accum_steps
            running_infonce        += infonce_val
            running_grounding      += grounding_val
            running_grounding_ema  += grounding_ema_val

            # --- debug: mem leak investigation (itertools.cycle GPU-batch retention bug) ---
            # if step % 7 == 0 or step % 15 == 0:
            #     allocated_gb = torch.cuda.memory_allocated(self.accelerator.device) / 1e9
            #     reserved_gb = torch.cuda.memory_reserved(self.accelerator.device) / 1e9
            #     kind = "grounding" if use_grounding else "spar"
            #     print(
            #         f"[mem] rank={self.accelerator.process_index} step={step} kind={kind} "
            #         f"pre-gc  allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB",
            #         flush=True,
            #     )
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     allocated_gb2 = torch.cuda.memory_allocated(self.accelerator.device) / 1e9
            #     reserved_gb2 = torch.cuda.memory_reserved(self.accelerator.device) / 1e9
            #     print(
            #         f"[mem] rank={self.accelerator.process_index} step={step} kind={kind} "
            #         f"post-gc allocated={allocated_gb2:.2f}GB reserved={reserved_gb2:.2f}GB",
            #         flush=True,
            #     )
            #
            #     from collections import Counter
            #     live: Counter = Counter()
            #     for obj in gc.get_objects():
            #         if torch.is_tensor(obj) and obj.is_cuda:
            #             live[(tuple(obj.shape), str(obj.dtype))] += obj.numel() * obj.element_size()
            #     top = sorted(live.items(), key=lambda kv: -kv[1])[:10]
            #     for shape_dtype, nbytes in top:
            #         print(
            #             f"[mem-shape] rank={self.accelerator.process_index} step={step} "
            #             f"kind={kind} shape={shape_dtype} total={nbytes / 1e9:.3f}GB",
            #             flush=True,
            #         )
            # --- end debug ---

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
                    # Raw grounding focal loss -- watch this to see whether
                    # grounding is actually improving. Averaged over log_every
                    # steps total, not just the ~1-in-grounding_ratio steps
                    # that were actually grounding, same as before.
                    "grounding_loss": running_grounding / self.log_every,
                    # EMA-normalized version -- this is what grounding_loss_weight
                    # actually multiplies (see _grounding_step). Stays ~O(1) by
                    # design, not a quality signal -- only meaningful for
                    # sanity-checking that the normalization itself is working.
                    "grounding_loss_ema_normalized": running_grounding_ema / self.log_every,
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
                                "train/grounding_loss_ema_normalized": entry["grounding_loss_ema_normalized"],
                                "train/lr":             entry["lr"],
                            },
                            step=step,
                        )
                running_loss = running_infonce = running_grounding = running_grounding_ema = 0.0

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
        proj_dim=f.get("proj_dim", 1024),
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
        # toy predictor path only (ignored when query_model_name != "toy")
        predictor_hidden_size=m.get("predictor_hidden_size", 512),
        predictor_layers=m.get("predictor_layers", 6),
        predictor_heads=m.get("predictor_heads", 8),
        predictor_ffn_mult=m.get("predictor_ffn_mult", 4),
        predictor_dropout=m.get("predictor_dropout", 0.0),
        # llama predictor path only (ignored when query_model_name == "toy")
        predictor_llama_layers=m.get("predictor_llama_layers", 8),
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
    num_frames = dcfg.get("grounding_num_frames", dcfg.get("num_frames"))
    batch_size = cfg["train"].get("grounding_batch_size", cfg["train"]["batch_size"])
    return _build_loader(manifests, dcfg, batch_size, num_frames=num_frames)


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
