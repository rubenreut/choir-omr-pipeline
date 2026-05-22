#!/usr/bin/env python3
"""ChoirSMT training script.

Architecture (Tier 2 — pretrained ConvNeXt vision encoder + from-scratch transformer decoder):

  Image (3, 1024, 1536)
     ↓
  ConvNeXt-Tiny (ImageNet pretrained)
     ↓ feature map (~32×48×768)
  Linear projection to d_model=512
     ↓ flatten + 2D positional encoding
  Transformer decoder (6 layers, 8 heads, d=512)
     ↓ cross-attends to image features
  Linear head → vocab_size logits
     ↓
  Token sequence (autoregressive)

Run:
    python3 train.py --manifest training.json --vocab vocab.json --epochs 50 --batch 8
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import ChoirOMRDataset, collate_pad


D_MODEL = 512
N_HEADS = 8
N_DECODER_LAYERS = 6
FFN_DIM = 2048


class ChoirSMT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 4096):
        super().__init__()
        # --- Visual encoder (ImageNet pretrained) ---
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        # Drop the classification head; keep feature extractor only
        self.encoder = nn.Sequential(*list(backbone.features))  # outputs (B, 768, H', W')
        self.proj_in = nn.Conv2d(768, D_MODEL, kernel_size=1)

        # 2D positional encoding (learnable, sized for max feature-map dims)
        self.pos_h = nn.Parameter(torch.randn(1, D_MODEL // 2, 64, 1) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, D_MODEL // 2, 1, 64) * 0.02)

        # --- Token embeddings + positional encoding for decoder ---
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.tok_pos = nn.Parameter(torch.randn(1, max_seq_len, D_MODEL) * 0.02)

        # --- Transformer decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_DECODER_LAYERS)

        # Output head
        self.head = nn.Linear(D_MODEL, vocab_size)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Image → memory tokens for decoder cross-attention."""
        feat = self.encoder(images)  # (B, 768, H', W')
        feat = self.proj_in(feat)    # (B, D, H', W')
        B, D, H, W = feat.shape
        # Add 2D positional encoding
        pos = torch.cat([
            self.pos_h[:, :, :H, :].expand(-1, -1, -1, W),
            self.pos_w[:, :, :, :W].expand(-1, -1, H, -1),
        ], dim=1)
        feat = feat + pos[:, :D]
        # Flatten to sequence
        return feat.flatten(2).transpose(1, 2)  # (B, H'×W', D)

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward pass. Returns logits of shape (B, T, V)."""
        memory = self.encode(images)
        T = tgt_tokens.size(1)
        tok = self.tok_emb(tgt_tokens) + self.tok_pos[:, :T]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=tok.device), diagonal=1
        )
        out = self.decoder(tok, memory, tgt_mask=causal_mask)
        return self.head(out)


def train(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] device: {device}")

    # Load vocab
    import json
    vocab = json.loads(Path(args.vocab).read_text())
    vocab_size = len(vocab)
    print(f"[train] vocab size: {vocab_size}")

    # Dataset
    dataset = ChoirOMRDataset(args.manifest, image_root=args.image_root)
    print(f"[train] dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_pad, pin_memory=True,
    )

    # Model
    model = ChoirSMT(vocab_size=vocab_size).to(device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # Optional mixed precision for speed on GPU
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for step, (imgs, toks) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            # Teacher-forcing: input = toks[:, :-1], target = toks[:, 1:]
            inp = toks[:, :-1]
            tgt = toks[:, 1:]

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(imgs, inp)
                    loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs, inp)
                loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)

        scheduler.step()
        avg = running_loss / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] avg_loss={avg:.4f} time={elapsed:.0f}s", flush=True)

        # Save checkpoint every epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "vocab_size": vocab_size,
        }, Path(args.output) / f"ckpt_epoch_{epoch:03d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON manifest of (image, tokens)")
    ap.add_argument("--vocab", required=True, help="vocab.json from tokenizer")
    ap.add_argument("--image-root", default=None, help="Root dir for images (default: manifest dir)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="./checkpoints")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
