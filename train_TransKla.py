
import os, sys, time, random, json, yaml
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# Perf knobs (Ampere)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- Amino acid utilities ----------
AMINO = "ACDEFGHIKLMNPQRSTVWYXBZJUO"  # include X + uncommon
AA2IDX = {aa: i + 1 for i, aa in enumerate(AMINO)}  # 0 = PAD
PAD_IDX = 0
X_IDX = AA2IDX.get("X", len(AA2IDX) + 1)

HYDRO = {  # Kyte-Doolittle
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
    'X': 0.0, 'B': -3.5, 'Z': -3.5, 'J': 0.0, 'U': 0.0, 'O': 0.0,
}
CHARGE = {'K': 1.0, 'R': 1.0, 'D': -1.0, 'E': -1.0, 'H': 0.1}

def aa_to_idx(a: str) -> int:
    return AA2IDX.get(a.upper(), X_IDX)

def seq_to_idx(seq: str) -> List[int]:
    return [aa_to_idx(c) for c in seq]

def read_fasta(path: str) -> List[Tuple[str, str, int, str]]:
    """Return (accession, seq, label, header) from >ACC|pos=...|{0,1}."""
    headers, seqs, buf = [], [], []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
                headers.append(ln[1:].split()[0])
            else:
                buf.append(ln)
        if buf:
            seqs.append("".join(buf))
    assert len(headers) == len(seqs), "FASTA parse mismatch"
    out = []
    for h, s in zip(headers, seqs):
        parts = h.split("|")
        acc = parts[0]
        lab = int(parts[2]) if len(parts) >= 3 and parts[2] in ("0", "1") else 1
        out.append((acc, s, lab, h))
    return out

def load_protein_map(csv_path: Optional[str]) -> Dict[str, str]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    need = {"Protein_accession", "Sequence"}
    if not need.issubset(df.columns):
        return {}
    return dict(zip(df["Protein_accession"].astype(str), df["Sequence"].astype(str)))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------- Dataset with augmentation ----------
class KlaDataset(Dataset):
    def __init__(self, fasta_path: str, protein_map: Dict[str, str],
                 global_mode: str = "simple", global_tokens: int = 8,
                 augment: bool = False, mask_prob: float = 0.08):
        rows = read_fasta(fasta_path)
        self.samples = rows
        self.protein_map = protein_map
        self.global_mode = global_mode if len(protein_map) > 0 else "none"
        self.global_tokens = global_tokens
        self.augment = augment
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _mask_seq(seq: str, p: float) -> str:
        """Randomly replace residues (except center index 20) with 'X'."""
        if len(seq) != 41 or p <= 0:
            return seq
        s = list(seq)
        for i in range(41):
            if i == 20:  # keep central K intact
                continue
            if random.random() < p:
                s[i] = 'X'
        return "".join(s)

    def _local_feats(self, seq: str):
        ids = seq_to_idx(seq)
        hydro = [HYDRO.get(c.upper(), 0.0) for c in seq]
        charge = [CHARGE.get(c.upper(), 0.0) for c in seq]
        return np.array(ids, np.int64), np.array(hydro, np.float32), np.array(charge, np.float32)

    def _global_tokens_simple(self, prot_seq: str, N: int):
        if not prot_seq:
            return np.zeros((N, 1), np.int64)
        idxs = seq_to_idx(prot_seq)
        L = len(idxs)
        segs = []
        for s in np.array_split(np.arange(L), N):
            segs.append([idxs[i] for i in s] if len(s) else [X_IDX])
        maxlen = max(len(s) for s in segs)
        arr = np.zeros((N, maxlen), np.int64)
        for i, s in enumerate(segs):
            arr[i, :len(s)] = s
        return arr

    def __getitem__(self, i):
        acc, seq, lab, hdr = self.samples[i]
        if self.augment:
            seq = self._mask_seq(seq, self.mask_prob)
        ids, hyd, chg = self._local_feats(seq)
        gtok = None
        if self.global_mode != "none":
            prot = self.protein_map.get(acc, "")
            gtok = self._global_tokens_simple(prot, self.global_tokens)
        return {
            "accession": acc,
            "local_ids": ids,
            "local_hydro": hyd[:, None],
            "local_charge": chg[:, None],
            "label": np.array([lab], np.float32),
            "global_chunks": gtok,
            "header": hdr,
        }

def collate(batch):
    B = len(batch)
    local_ids = torch.from_numpy(np.stack([b["local_ids"] for b in batch], 0))  # [B,41]
    local_aux = torch.from_numpy(np.concatenate([b["local_hydro"] for b in batch], 0)).view(B, 41, 1)
    local_chg = torch.from_numpy(np.concatenate([b["local_charge"] for b in batch], 0)).view(B, 41, 1)
    labels = torch.from_numpy(np.stack([b["label"] for b in batch], 0))[:, 0]

    global_chunks = None
    if batch[0]["global_chunks"] is not None:
        N = batch[0]["global_chunks"].shape[0]
        maxlen = max(b["global_chunks"].shape[1] for b in batch)
        gc = np.zeros((B, N, maxlen), np.int64)
        mask = np.zeros((B, N, maxlen), np.float32)
        for i, b in enumerate(batch):
            gl = b["global_chunks"]
            gc[i, :gl.shape[0], :gl.shape[1]] = gl
            mask[i, :gl.shape[0], :gl.shape[1]] = (gl > 0).astype(np.float32)
        global_chunks = (torch.from_numpy(gc), torch.from_numpy(mask))
    return {
        "local_ids": local_ids,
        "local_aux": torch.cat([local_aux, local_chg], dim=2),  # [B,41,2]
        "labels": labels.float(),
        "global": global_chunks,
        "headers": [b["header"] for b in batch],
        "accessions": [b["accession"] for b in batch],
    }

# ---------- Stochastic depth ----------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

# ---------- Transformer building blocks ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, mlp_ratio=2.0, dropout=0.2, droppath=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dp1 = DropPath(droppath)
        self.dp2 = DropPath(droppath)

    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dp1(h)
        h = self.mlp(self.ln2(x))
        x = x + self.dp2(h)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.2, droppath=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lnq = nn.LayerNorm(d_model)
        self.lnkv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.dp = DropPath(droppath)

    def forward(self, q, k, v, key_padding_mask=None):
        h, _ = self.attn(self.lnq(q), self.lnkv(k), self.lnkv(v),
                         key_padding_mask=key_padding_mask, need_weights=False)
        return q + self.dp(self.ff(h))

class TransKla(nn.Module):
    def __init__(self, d_model=256, nhead=4, n_layers=6, d_emb=128,
                 use_global=True, global_tokens=8, dropout=0.2, droppath=0.1):
        super().__init__()
        self.use_global = use_global
        vocab = len(AA2IDX) + 1
        self.token_emb = nn.Embedding(vocab, d_emb, padding_idx=PAD_IDX)
        self.aux_proj = nn.Linear(2, d_emb)  # hydro/charge -> d_emb
        self.pos_emb = nn.Embedding(41, d_emb)  # positions [0..40]
        self.local_in = nn.Linear(d_emb * 3, d_model)  # concat(token, aux, pos)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # linearly scaled droppath per block
        self.local_stack = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio=2.0,
                             dropout=dropout, droppath=droppath * float(i) / max(1, (n_layers - 1)))
            for i in range(n_layers)
        ])

        if self.use_global:
            self.global_token_proj = nn.Linear(d_emb, d_model)
            self.cross1 = CrossAttention(d_model, nhead, dropout, droppath / 2.0)
            self.cross2 = CrossAttention(d_model, nhead, dropout, droppath / 2.0)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, local_ids, local_aux, global_chunks=None):
        B, L = local_ids.shape
        tok = self.token_emb(local_ids)  # [B,L,d_emb]
        aux = self.aux_proj(local_aux)   # [B,L,d_emb]
        pos_idx = torch.arange(L, device=local_ids.device)[None, :]
        pos = self.pos_emb(pos_idx.repeat(B, 1))  # [B,L,d_emb]
        x = torch.cat([tok, aux, pos], dim=-1)
        x = self.local_in(x)  # [B,L,d_model]

        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]
        x = torch.cat([cls, x], dim=1)          # [B,1+L,d_model]
        for blk in self.local_stack:
            x = blk(x)
        cls_vec = x[:, 0:1, :]  # [B,1,d_model]

        if self.use_global and global_chunks is not None:
            ids, mask = global_chunks  # ids: [B,N,Ls], mask: [B,N,Ls]
            g = self.token_emb(ids)                     # [B,N,Ls,d_emb]
            m = (mask > 0).float()                      # [B,N,Ls]
            msum = m.sum(dim=2, keepdim=True).clamp_min(1.0)
            g = (g * m.unsqueeze(-1)).sum(dim=2) / msum # [B,N,d_emb]
            g = self.global_token_proj(g)               # [B,N,d_model]
            cls_vec = self.cross1(cls_vec, g, g)
            cls_vec = self.cross2(cls_vec, g, g)

        logit = self.head(cls_vec).squeeze(1).squeeze(1)  # [B]
        return logit

# ---------- EMA ----------
class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.995, device=None):
        self.decay = decay
        self.shadow = {k: v.detach().clone().to(device or v.device) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def load_to(self, model: nn.Module, device=None):
        model.load_state_dict({k: v.to(device or v.device) for k, v in self.shadow.items()}, strict=True)

# ---------- Metrics ----------
def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def metrics_from_scores(y_true, y_score, thresh=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thresh).astype(int)
    out = {}
    try:
        out["auroc"] = roc_auc_score(y_true, y_score)
    except:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = average_precision_score(y_true, y_score)
    except:
        out["auprc"] = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = r  # sensitivity = recall
    acc = float((y_pred == y_true).mean())
    out.update(dict(precision=p, recall=r, f1=f1, acc=acc, specificity=spec, sensitivity=sens,
                    tp=tp, tn=tn, fp=fp, fn=fn, threshold=thresh))
    out["mcc"] = matthews_corrcoef(y_true, y_pred) if (tp + tn + fp + fn) > 0 else float("nan")
    return out

def best_mcc_threshold(y_true, y_score):
    """Find threshold that maximizes Matthews Correlation Coefficient"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_mcc = -1.0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (np.asarray(y_score) >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    return best_thresh, best_mcc

def best_f1_threshold(y_true, y_score):
    ts = np.linspace(0, 1, 101)
    best = (0.0, 0.5)
    for t in ts:
        f1 = f1_score(y_true, (np.asarray(y_score) >= t).astype(int), zero_division=0)
        if f1 > best[0]:
            best = (f1, t)
    return best[1], best[0]

# ---------- Warmup + Cosine LR ----------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            s = epoch / max(1, self.warmup_epochs)
            return [base_lr * s for base_lr in self.base_lrs]
        e = epoch - self.warmup_epochs
        T = max(1, self.max_epochs - self.warmup_epochs)
        cos = 0.5 * (1 + np.cos(np.pi * e / T))
        return [self.min_lr + (base_lr - self.min_lr) * cos for base_lr in self.base_lrs]

# ---------- Train/Eval loops ----------
def run_epoch(model, loader, device, optimizer=None, scaler=None, ema: Optional[ModelEMA] = None):
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    y_true = []
    y_score = []
    bce = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(loader, 1):
        x_ids = batch["local_ids"].to(device, non_blocking=True)
        x_aux = batch["local_aux"].to(device, non_blocking=True)
        glb = None
        if batch["global"] is not None:
            ids, mask = batch["global"]
            glb = (ids.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        y = batch["labels"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(x_ids, x_aux, glb)
                    loss = bce(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_ids, x_aux, glb)
                loss = bce(logits, y)
                loss.backward()
                optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad():
                logits = model(x_ids, x_aux, glb)
                loss = bce(logits, y)

        total_loss += loss.item() * x_ids.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_score.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_score

def make_loader(fa, protein_map, batch_size, shuffle, num_workers, global_mode, global_tokens,
                augment, mask_prob):
    ds = KlaDataset(fa, protein_map=protein_map, global_mode=global_mode,
                    global_tokens=global_tokens, augment=augment, mask_prob=mask_prob)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate, drop_last=False)

# ---------- Main ----------
def main(cfg: dict):
    os.makedirs(cfg["outdir"], exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # Load protein map
    prot_map = load_protein_map(cfg.get("protein_csv", ""))
    global_mode = "simple" if cfg.get("use_global", True) and len(prot_map) > 0 else "none"
    if len(prot_map) == 0 and cfg.get("use_global", True):
        print("[WARN] protein_csv missing/empty; disabling global branch.")

    print("[INFO] Building dataloaders...")
    train_loader = make_loader(
        cfg["train_fa"], prot_map, cfg["batch_size"], True, cfg["num_workers"],
        global_mode, cfg["global_tokens"], augment=True, mask_prob=cfg["mask_prob"],
    )
    val_loader = make_loader(
        cfg["val_fa"], prot_map, cfg["batch_size"], False, cfg["num_workers"],
        global_mode, cfg["global_tokens"], augment=False, mask_prob=0.0,
    )
    test_loader = make_loader(
        cfg["test_fa"], prot_map, cfg["batch_size"], False, cfg["num_workers"],
        global_mode, cfg["global_tokens"], augment=False, mask_prob=0.0,
    )

    print("[INFO] Creating model...")
    model = TransKla(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_layers"],
        d_emb=cfg["d_emb"],
        use_global=(global_mode != "none"),
        global_tokens=cfg["global_tokens"],
        dropout=cfg["dropout"],
        droppath=cfg["droppath"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    sched = WarmupCosineLR(opt, warmup_epochs=cfg["warmup_epochs"], max_epochs=cfg["epochs"],
                           min_lr=cfg["min_lr"])

    ema = ModelEMA(model, decay=0.995, device=device)

    best_auprc = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print("[INFO] Training...")
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_s = run_epoch(model, train_loader, device, optimizer=opt,
                                        scaler=scaler, ema=ema)

        # Evaluate with EMA weights
        current_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.load_to(model, device=device)
        va_loss, va_y, va_s = run_epoch(model, val_loader, device, optimizer=None)
        model.load_state_dict(current_state)

        tr_m = metrics_from_scores(tr_y, tr_s, 0.5)
        va_m = metrics_from_scores(va_y, va_s, 0.5)

        sched.step()

        improved = (va_m["auprc"] > best_auprc + cfg["min_delta"])
        if improved:
            best_auprc = va_m["auprc"]
            best_epoch = epoch
            ema_snapshot = ema.shadow
            best_state = {k: v.detach().cpu() for k, v in ema_snapshot.items()}
            epochs_no_improve = 0
            torch.save(best_state, os.path.join(cfg["outdir"], "klaformer_best.pt"))
            print(f"[INFO] NEW BEST at epoch {epoch:02d}: val AUPRC {best_auprc:.4f} -- checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No new best (delta AUPRC {va_m['auprc'] - best_auprc:+.4f}); "
                  f"patience {cfg['patience'] - epochs_no_improve}/{cfg['patience']}")

        dt = time.time() - t0
        print(f"[EPOCH {epoch:02d}] {dt:.1f}s "
              f"| train loss {tr_loss:.4f} AUPRC {tr_m['auprc']:.4f} AUROC {tr_m['auroc']:.4f} "
              f"F1 {tr_m['f1']:.4f} MCC {tr_m['mcc']:.4f} "
              f"| val loss {va_loss:.4f} AUPRC {va_m['auprc']:.4f} AUROC {va_m['auroc']:.4f} "
              f"F1 {va_m['f1']:.4f} MCC {va_m['mcc']:.4f}")

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
        for k, v in tr_m.items():
            row[f"train_{k}"] = v
        row["val_loss"] = va_loss
        for k, v in va_m.items():
            row[f"val_{k}"] = v
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(cfg["outdir"], "history.csv"), index=False)

        if epochs_no_improve >= cfg["patience"]:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # Restore best EMA weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
        print(f"[INFO] Restored EMA-best model from epoch {best_epoch}.")
    else:
        print("[WARN] No improvement recorded; evaluating with last EMA snapshot.")
        ema.load_to(model, device=device)

    print("[INFO] Calibrating threshold t* on validation set...")
    _, val_y, val_s = run_epoch(model, val_loader, device, optimizer=None)
    t_star, mcc_star = best_mcc_threshold(val_y, val_s)
    print(f"[INFO] t* (best-MCC) = {t_star:.3f}, MCC = {mcc_star:.4f}")
    val_metrics = metrics_from_scores(val_y, val_s, t_star)

    print("[INFO] Testing...")
    te_loss, te_y, te_s = run_epoch(model, test_loader, device, optimizer=None)
    test_metrics = metrics_from_scores(te_y, te_s, t_star)

    pd.DataFrame({"y_true": te_y, "y_score": te_s}).to_csv(
        os.path.join(cfg["outdir"], "test_predictions.csv"), index=False
    )

    final_rows = [
        {"split": "val", "loss": float("nan"), **val_metrics},
        {"split": "test", "loss": float(te_loss), **test_metrics},
    ]
    pd.DataFrame(final_rows).to_csv(os.path.join(cfg["outdir"], "final_metrics.csv"), index=False)

    with open(os.path.join(cfg["outdir"], "config.json"), "w") as f:
        # make sure all values are JSON serializable (e.g., convert Path objects)
        json.dump(cfg, f, indent=2, default=str)

    print(f"[VAL ] AUPRC {val_metrics['auprc']:.4f} AUROC {val_metrics['auroc']:.4f} "
          f"Acc {val_metrics['acc']:.4f} F1 {val_metrics['f1']:.4f} "
          f"Prec {val_metrics['precision']:.4f} Rec/Sen {val_metrics['sensitivity']:.4f} "
          f"Spec {val_metrics['specificity']:.4f} MCC {val_metrics['mcc']:.4f}")
    print(f"[TEST] loss {te_loss:.4f} AUPRC {test_metrics['auprc']:.4f} AUROC {test_metrics['auroc']:.4f} "
          f"Acc {test_metrics['acc']:.4f} F1@t* {test_metrics['f1']:.4f} "
          f"Prec {test_metrics['precision']:.4f} Rec/Sen {test_metrics['sensitivity']:.4f} "
          f"Spec {test_metrics['specificity']:.4f} MCC {test_metrics['mcc']:.4f}")
    print("[DONE]")

# ---------- Config loading ----------
def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and fill in default values for missing keys.
    Also ensures that if 'protein_csv' is missing/empty, global branch is disabled.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Essential defaults (can be overridden by config.yaml)
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "outdir": "runs/klaformer",
        "seed": 42,
        "num_workers": 4,
        "min_delta": 0.002,
        "mask_prob": 0.08,
        "dropout": 0.2,
        "droppath": 0.1,
        "d_model": 256,
        "nhead": 4,
        "n_layers": 6,
        "d_emb": 128,
        "use_global": True,
        "global_tokens": 8,
        "batch_size": 32,
        "epochs": 40,
        "lr": 2e-4,
        "weight_decay": 1e-2,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "patience": 10,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # Disable global branch if protein CSV is not provided
    if not cfg.get("protein_csv", ""):
        cfg["use_global"] = False

    return cfg

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    main(cfg)