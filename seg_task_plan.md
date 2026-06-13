# Plan: EmbodiedScan grounding + VL-JEPA seg head + mixed training

## Context
Add grounding capability to existing FusionGVJEPA pipeline.
Current predictor collapses spatial dim (1369→1, mean-pool) before predictor → no spatial output.
Fix: keep spatial skip from x_encoder, cross-attend predictor scene tokens back to spatial patches → seg logits (B,S,P).
Mixed train: SPAR-7M QA (InfoNCE) + EmbodiedScan grounding (BCE), separate DataLoaders, alternating in fit().

---

## File 1: `dataset/download_embodiedscan.sh` (NEW)

Download EmbodiedScan annotation PKL from HuggingFace `OpenRobotLab/EmbodiedScan`.

```bash
#!/usr/bin/env bash
# Downloads EmbodiedScan annotation PKLs only.
# Images (ScanNet/HM3D/MP3D) require separate data agreements — see README.
set -euo pipefail

REPO_ID="OpenRobotLab/EmbodiedScan"
OUTPUT_DIR="${1:-./source_data/embodiedscan}"
mkdir -p "$OUTPUT_DIR"

FILES=(
  "embodiedscan_infos_train_full.pkl"
  "embodiedscan_infos_val_full.pkl"
)

for f in "${FILES[@]}"; do
  wget -c "https://huggingface.co/datasets/${REPO_ID}/resolve/main/${f}" -P "$OUTPUT_DIR"
done

echo "Done. Images need ScanNet/HM3D/MP3D separate download."
```

**Note:** exact filenames depend on EmbodiedScan HF repo structure — verify at
`https://huggingface.co/datasets/OpenRobotLab/EmbodiedScan/tree/main` before finalizing.

---

## File 2: `dataset/convert_embodiedscan.py` (NEW)

Convert EmbodiedScan PKL → JSONL matching existing schema + `task_type`/`boxes`.

Key shapes / coordinate math:
- PKL `data_list[i]`: has `images` dict (cam_type → `{img_path, cam2img (3×3), cam2global (4×4)}`), `ann_info.gt_bboxes_3d` (N×9, xyzlwhyaw...)
- 3D bbox → 8 corners (world) → cam via `inv(cam2global)` → image via `cam2img`
- Clip to image bounds, skip if all corners behind camera (depth ≤ 0)
- Output box: `[x1,y1,x2,y2]` float, or `null` if not visible in that view

```python
# dataset/convert_embodiedscan.py
def bbox9_to_corners(bbox9):
    """xyzlwhyaw... → (8,3) world corners"""

def project_corners(corners_w, cam2global, cam2img, img_hw):
    """world (8,3) → [x1,y1,x2,y2] image bbox or None"""

def convert(pkl_path, image_root, output_jsonl, max_views=4):
    data = pickle.load(open(pkl_path,"rb"))
    for sample in data["data_list"]:
        cam_views = list(sample["images"].values())[:max_views]
        images = [str(Path(image_root)/v["img_path"]) for v in cam_views]
        for ann in sample["ann_info"]["grounding"]:   # grounding sub-key
            boxes = [project_corners(..., v["cam2global"], v["cam2img"], hw)
                     for v in cam_views]
            if not any(b is not None for b in boxes):
                continue
            row = {
                "images": images,
                "query": ann["description"],
                "target": ann["description"],
                "task_type": "grounding",
                "boxes": boxes,
                "source": "embodiedscan",
            }
            out.write(json.dumps(row) + "\n")
```

CLI args: `--pkl`, `--image-root`, `--output`, `--max-views` (default 4), `--split` (train/val).

---

## File 3: `fusion_gv/gvjepa.py` (MODIFY)

### 3a. `GVJEPAConfig` — add 2 fields
```python
enable_grounding: bool = False
seg_heads: int = 8
```

### 3b. `FusionGVJEPA.__init__` — add grounding head (conditional)
After existing modules, append:
```python
self.seg_cross_attn = None
self.seg_head = None
if config.enable_grounding:
    D_f = config.fusion.visual_dim   # 3072 (concat) or d_fusion (cross_attn)
    h   = config.predictor_hidden_size  # 512
    self.seg_cross_attn = nn.MultiheadAttention(
        embed_dim=D_f, num_heads=config.seg_heads,
        kdim=h, vdim=h, batch_first=True,
    )
    self.seg_head = nn.Linear(D_f, 1)
```

### 3c. `_pool_visual` — return skip
```python
def _pool_visual(self, images_vggt, images_jepa):
    feats = self.x_encoder(images_vggt, images_jepa)
    feat  = feats[self.config.use_fusion_level]   # (B, S, P, D_f)  P=1369
    return feat.mean(dim=2), feat                  # (B,S,D_f), (B,S,P,D_f)
```

### 3d. `predict_embedding` — unpack skip (ignore it)
Change: `vis = self._pool_visual(...)` → `vis_pooled, _ = self._pool_visual(...)`; rename `vis` to `vis_pooled`.

### 3e. `predict_grounding` — new method
```python
def predict_grounding(self, images_vggt, images_jepa, queries):
    assert self.seg_cross_attn is not None, "enable_grounding=False"
    device = images_vggt.device
    B, S   = images_vggt.shape[:2]

    vis_pooled, vis_skip = self._pool_visual(images_vggt, images_jepa)
    vis = self.vis_proj(vis_pooled)          # (B, S, h)
    # query stream (same as predict_embedding)
    q_tok = self._tokenize(self.query_tokenizer, queries, self.config.max_query_tokens, device)
    q_emb = self.query_in_proj(
        self.query_encoder.get_input_embeddings()(q_tok["input_ids"])
    )                                         # (B, L, h)
    x = torch.cat([vis, q_emb], dim=1)
    vis_mask = torch.ones(B, S, device=device, dtype=q_tok["attention_mask"].dtype)
    x = self.predictor(x, src_key_padding_mask=(
        torch.cat([vis_mask, q_tok["attention_mask"]], dim=1) == 0
    ))                                        # (B, S+L, h)

    vis_pred = x[:, :S, :]                   # (B, S, h) — query-conditioned scene tokens
    P   = vis_skip.shape[2]                  # 1369
    D_f = vis_skip.shape[3]                  # 3072
    skip_flat = vis_skip.reshape(B*S, P, D_f)
    kv_flat   = vis_pred.reshape(B*S, 1, -1)  # (B*S, 1, h)
    out, _    = self.seg_cross_attn(skip_flat, kv_flat, kv_flat)  # (B*S, P, D_f)
    logits    = self.seg_head(out).squeeze(-1)                     # (B*S, P)
    return logits.reshape(B, S, P)                                 # (B, S, P)
```

### 3f. `forward` — accept `task_type`
```python
def forward(self, images_vggt, images_jepa, queries,
            targets=None, task_type="qa"):
    if task_type == "grounding":
        return {"seg_logits": self.predict_grounding(images_vggt, images_jepa, queries)}
    pred   = self.predict_embedding(images_vggt, images_jepa, queries)
    target = self.encode_target(targets, images_vggt.device)
    return {"pred": pred, "target": target}
```

---

## File 4: `fusion_gv/gvjepa_trainer.py` (MODIFY)

### 4a. `GVJEPADataset.__getitem__` — read grounding fields
```python
return {
    "image_paths": image_paths,
    "query":       row.get("query", ""),
    "target":      row.get("target", ""),
    "task_type":   row.get("task_type", "qa"),
    "boxes":       row.get("boxes"),   # list[bbox|null] per view, or None
}
```

### 4b. New helper `boxes_to_patch_mask`
```python
def boxes_to_patch_mask(boxes, S, patch_size=14, img_size=518):
    """boxes: list[S] of [x1,y1,x2,y2]|None → (S, P) float tensor"""
    grid = img_size // patch_size   # 37
    P    = grid * grid              # 1369
    mask = torch.zeros(S, P)
    for i, box in enumerate(boxes or []):
        if box is None:
            continue
        x1,y1,x2,y2 = box
        pj1 = max(0, int(x1/patch_size));  pj2 = min(grid, int(x2/patch_size)+1)
        pi1 = max(0, int(y1/patch_size));  pi2 = min(grid, int(y2/patch_size)+1)
        g = torch.zeros(grid, grid)
        g[pi1:pi2, pj1:pj2] = 1.0
        mask[i] = g.flatten()
    return mask
```

### 4c. `gvjepa_collate` — handle grounding
Add after existing stacking:
```python
task_type = batch[0]["task_type"]
result = {"images_vggt":..., "images_jepa":..., "query":..., "target":..., "task_type": task_type}
if task_type == "grounding":
    S = images_vggt.shape[1]
    result["patch_labels"] = torch.stack(
        [boxes_to_patch_mask(s["boxes"], S) for s in batch]
    )   # (B, S, P)
return result
```

### 4d. `GVJEPATrainer.__init__` — new params
```python
grounding_loader: Optional[DataLoader] = None,
grounding_loss_weight: float = 1.0,
grounding_pos_weight: float = 10.0,
grounding_ratio: int = 3,   # 1 grounding step per N qa steps
```

### 4e. `GVJEPATrainer.fit()` — alternating loss
```python
import itertools, torch.nn.functional as F

grounding_iter = itertools.cycle(self.grounding_loader) if self.grounding_loader else None

# inside step loop, after qa forward:
out  = self.model(..., task_type="qa")
loss = bidirectional_infonce(out["pred"], out["target"], self.temperature)

if grounding_iter and (step % self.grounding_ratio == 0):
    g_batch = next(grounding_iter)
    g_vggt  = g_batch["images_vggt"].to(device, non_blocking=True)
    g_jepa  = g_batch["images_jepa"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=self.autocast_dtype):
        g_out  = self.model(g_vggt, g_jepa, g_batch["query"], task_type="grounding")
        g_loss = F.binary_cross_entropy_with_logits(
            g_out["seg_logits"],
            g_batch["patch_labels"].to(device),
            pos_weight=torch.tensor(self.grounding_pos_weight, device=device),
        )
    loss = loss + self.grounding_loss_weight * g_loss
```

### 4f. `build_loader_from_config` — add grounding loader
Support new YAML key `grounding_manifests` → build second DataLoader same way as QA, return tuple.

---

## File 5: `fusion_gv/debug_one_batch.py` (NEW)

Minimal end-to-end smoke test with toy tensors (no real data needed):

```python
#!/usr/bin/env python3
"""One-batch smoke test for QA + grounding forward + backward."""
import torch, torch.nn.functional as F
from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig
from fusion_gv.config import FusionConfig
from fusion_gv.gvjepa_trainer import boxes_to_patch_mask, bidirectional_infonce

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, S, P = 2, 4, 1369

cfg = GVJEPAConfig(
    fusion=FusionConfig(x_encoder_type="vjepa_only"),  # skip VGGT for speed
    enable_grounding=True,
    predictor_hidden_size=128,
    predictor_layers=2,
    shared_embed_dim=64,
)
model = FusionGVJEPA(cfg).to(DEVICE)

# Toy inputs (bypass real encoders with mock)
# ... create (B,S,3,518,518) vggt and (B*S,3,1,384,384) jepa tensors ...

# --- QA forward ---
out_qa = model(vggt, jepa, queries=["q"]*B, targets=["t"]*B, task_type="qa")
assert out_qa["pred"].shape  == (B, 64), out_qa["pred"].shape
assert out_qa["target"].shape== (B, 64)
loss_qa = bidirectional_infonce(out_qa["pred"], out_qa["target"])
loss_qa.backward()
print(f"QA loss: {loss_qa.item():.4f}  ✓")
model.zero_grad()

# --- Grounding forward ---
out_gr = model(vggt, jepa, queries=["q"]*B, task_type="grounding")
assert out_gr["seg_logits"].shape == (B, S, P), out_gr["seg_logits"].shape
labels = torch.zeros(B, S, P, device=DEVICE)
labels[:, 0, 100:120] = 1.0
loss_gr = F.binary_cross_entropy_with_logits(out_gr["seg_logits"], labels,
          pos_weight=torch.tensor(10.0, device=DEVICE))
loss_gr.backward()
print(f"Grounding loss: {loss_gr.item():.4f}  shape {out_gr['seg_logits'].shape}  ✓")

# --- boxes_to_patch_mask ---
boxes = [[0,0,112,112], None, [200,200,400,400], None]
mask  = boxes_to_patch_mask(boxes, S=4)
assert mask.shape == (4, 1369)
assert mask[0].sum() > 0 and mask[1].sum() == 0
print("boxes_to_patch_mask  ✓")
print("All checks passed.")
```

---

## YAML config additions

```yaml
model:
  enable_grounding: true
  seg_heads: 8

data:
  train_manifests:
    - data/spar_sftqa_train.jsonl
  grounding_manifests:
    - data/embodiedscan_train.jsonl

train:
  grounding_loss_weight: 1.0
  grounding_pos_weight: 10.0
  grounding_ratio: 3        # 1 grounding step per 3 QA steps
```

---

## Execution order

1. `bash dataset/download_embodiedscan.sh`
2. `python dataset/convert_embodiedscan.py --pkl ... --image-root ... --output data/embodiedscan_train.jsonl`
3. Modify `gvjepa.py` (3a–3f)
4. Modify `gvjepa_trainer.py` (4a–4f)
5. `python fusion_gv/debug_one_batch.py` — all ✓ before real training
6. Launch training with updated YAML

## Verification

- `debug_one_batch.py` prints QA loss, grounding loss, seg_logits shape, boxes_to_patch_mask shape — all pass
- `seg_logits.shape == (B, S, 1369)` confirmed in debug
- `patch_labels.sum() > 0` for visible views, `== 0` for null boxes
- Training log shows both `loss_qa` and `loss_grounding` decreasing
