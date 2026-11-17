"""Experiment 1: mini-batch gradient noise vs batch size.

For a fixed parameter state, compare the mini-batch gradient g_B with the
full gradient g over a fixed subset S of the training set, and measure

    noise^2(B) = E || g_B - g ||^2

Theory says E[g_B] = g (unbiased) and noise^2(B) = tr(Sigma) / B, so a
log-log plot of noise^2 vs B should have slope -1. The fit also gives an
estimate of the gradient noise scale C = tr(Sigma) and the critical batch
size B* = C / ||g||^2.

Usage:
    python exp1_gradient_noise.py
    python exp1_gradient_noise.py --warmup-epochs 1
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Gradient noise vs batch size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--subset-size", type=int, default=5000,
                   help="|S|: fixed subset used for full and mini-batch gradients")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--num-batches", type=int, default=200,
                   help="number of random batches sampled per batch size")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="train this many epochs before measuring the gradient noise")
    p.add_argument("--track-b", type=int, default=64,
                   help="batch size whose gradient coordinates are kept for the histogram")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--plot-dir", default="plots")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mlp(hidden, in_dim=784, num_classes=10):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def mnist_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def load_fixed_subset(args):
    """Return the fixed subset S as flat tensors (n, 784) and (n,)."""
    ds = datasets.MNIST(args.data_dir, train=True, download=True,
                        transform=mnist_transform())
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=gen)[: args.subset_size]
    x = torch.stack([ds[i][0] for i in idx]).view(len(idx), -1)
    y = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return x, y


def warmup(model, args, device):
    ds = datasets.MNIST(args.data_dir, train=True, download=True,
                        transform=mnist_transform())
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=gen)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(1, args.warmup_epochs + 1):
        for xb, yb in loader:
            xb = xb.to(device).view(xb.size(0), -1)
            yb = yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"warmup epoch {epoch}/{args.warmup_epochs} done")


def flatten_grad(model):
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def zero_grads(model):
    for p in model.parameters():
        p.grad = None


def full_gradient(model, x, y, chunk=1000):
    """Gradient of the mean loss over the whole subset S."""
    zero_grads(model)
    n = x.size(0)
    for i in range(0, n, chunk):
        xb = x[i:i + chunk]
        yb = y[i:i + chunk]
        loss = F.cross_entropy(model(xb), yb, reduction="sum") / n
        loss.backward()
    return flatten_grad(model)


def batch_gradient(model, xb, yb):
    """Gradient of the mean loss over one mini-batch (parameters not updated)."""
    zero_grads(model)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    return flatten_grad(model)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    stage = "init" if args.warmup_epochs == 0 else f"epoch{args.warmup_epochs}"
    print(f"device={device}  stage={stage}  args={vars(args)}")

    set_seed(args.seed)
    model = build_mlp(args.hidden).to(device)
    if args.warmup_epochs > 0:
        warmup(model, args, device)
    model.eval()

    x, y = load_fixed_subset(args)
    x, y = x.to(device), y.to(device)
    n = x.size(0)
    print(f"fixed subset |S|={n}")

    t0 = time.time()
    g = full_gradient(model, x, y).double()
    g_norm_sq = g.pow(2).sum().item()
    print(f"full gradient computed in {time.time() - t0:.1f}s  "
          f"||g||^2={g_norm_sq:.6e}")

    # Coordinates kept for the CLT histogram (evenly spaced across the vector).
    step = max(1, g.numel() // 1000)
    track_idx = torch.arange(0, g.numel(), step)[:1000].to(device)

    rows = []
    noise_by_b = {}
    tracked_by_b = None

    for B in args.batch_sizes:
        gen = torch.Generator().manual_seed(args.seed * 10007 + B)
        sum_g = torch.zeros_like(g)
        noise_sq_sum = 0.0
        tracked = []

        tb = time.time()
        for _ in range(args.num_batches):
            idx = torch.randint(0, n, (B,), generator=gen)
            xb, yb = x[idx], y[idx]
            gb = batch_gradient(model, xb, yb).double()
            sum_g += gb
            noise_sq_sum += (gb - g).pow(2).sum().item()
            if B == args.track_b:
                tracked.append(gb[track_idx].clone())

        mean_g = sum_g / args.num_batches
        rel_err = ((mean_g - g).pow(2).sum().sqrt()
                   / g.pow(2).sum().sqrt()).item()
        noise_sq = noise_sq_sum / args.num_batches
        noise_by_b[B] = noise_sq
        if B == args.track_b:
            tracked_by_b = torch.stack(tracked).cpu()

        row = {
            "stage": stage,
            "seed": args.seed,
            "batch_size": B,
            "num_batches": args.num_batches,
            "noise_sq": noise_sq,
            "noise_norm": float(np.sqrt(noise_sq)),
            "C_est": B * noise_sq,
            "mean_grad_rel_err": rel_err,
            "log10_B": float(np.log10(B)),
            "log10_noise_sq": float(np.log10(noise_sq)),
        }
        rows.append(row)
        print(f"B={B:4d}  noise^2={noise_sq:.6e}  "
              f"std={np.sqrt(noise_sq):.3e}  "
              f"C_est={B * noise_sq:.6e}  "
              f"rel_err={rel_err:.4f}  ({time.time() - tb:.1f}s)")

    # log-log fit: log10(noise^2) = intercept + slope * log10(B)
    Bs = np.array(args.batch_sizes, dtype=float)
    logB = np.log10(Bs)
    logN = np.log10(np.array([noise_by_b[B] for B in args.batch_sizes]))
    slope, intercept = np.polyfit(logB, logN, 1)
    pred = slope * logB + intercept
    ss_res = float(((logN - pred) ** 2).sum())
    ss_tot = float(((logN - logN.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    C = float(10 ** intercept)          # tr(Sigma)
    B_star = C / g_norm_sq

    print(f"\nfit: slope={slope:.4f}  R^2={r2:.6f}")
    print(f"C=tr(Sigma) ~ {C:.6e}   ||g||^2={g_norm_sq:.6e}   "
          f"B*={B_star:.2f}")

    # ---- save CSV and summary JSON ----
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"exp1_{stage}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": stage,
        "seed": args.seed,
        "hidden": args.hidden,
        "subset_size": n,
        "num_batches": args.num_batches,
        "batch_sizes": args.batch_sizes,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "C_tr_sigma": C,
        "g_norm_sq": g_norm_sq,
        "B_star": float(B_star),
        "noise_sq_by_B": {str(B): noise_by_b[B] for B in args.batch_sizes},
        "mean_grad_rel_err_by_B": {
            str(r["batch_size"]): r["mean_grad_rel_err"] for r in rows
        },
    }
    json_path = os.path.join(args.out_dir, f"exp1_{stage}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    tracked_path = None
    if tracked_by_b is not None:
        tracked_path = os.path.join(args.out_dir, f"exp1_{stage}_tracked.npz")
        np.savez(
            tracked_path,
            tracked=tracked_by_b.numpy(),
            g_tracked=g[track_idx].detach().cpu().numpy(),
            track_idx=track_idx.detach().cpu().numpy(),
        )

    # ---- plot ----
    os.makedirs(args.plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.scatter(Bs, [noise_by_b[B] for B in args.batch_sizes],
               color="#2563eb", zorder=3, label="measured")
    fit_x = np.logspace(np.log10(Bs.min()), np.log10(Bs.max()), 100)
    ax.plot(fit_x, 10 ** (intercept + slope * np.log10(fit_x)),
            color="#dc2626", lw=2,
            label=f"fit: slope={slope:.3f}, $R^2$={r2:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(r"noise$^2$ = E$\Vert g_B - g\Vert^2$")
    ax.set_title(f"Gradient noise vs batch size ({stage})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.text(0.03, 0.05,
            f"$B^*$ ≈ {B_star:.1f}\n$\\Vert g\\Vert^2$ = {g_norm_sq:.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax = axes[1]
    if tracked_by_b is not None:
        vals = tracked_by_b.numpy()
        var_per_coord = vals.var(axis=0, ddof=1)
        k = int(np.argmax(var_per_coord))
        sample = vals[:, k]
        mu, sigma = float(sample.mean()), float(sample.std(ddof=1))
        g_val = float(g[track_idx[k]].item())
        ax.hist(sample, bins=24, density=True, alpha=0.65,
                color="#2563eb", label="mini-batch gradients")
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2 * np.pi)),
                color="#dc2626", lw=2, label="normal fit")
        ax.axvline(g_val, color="black", ls="--", lw=1.5,
                   label="full-batch gradient")
        ax.set_xlabel("gradient value at one coordinate")
        ax.set_ylabel("density")
        ax.set_title(f"CLT check at B={args.track_b} (coordinate {int(track_idx[k])})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"no tracked coords for B={args.track_b}",
                ha="center", va="center")

    fig.tight_layout()
    plot_path = os.path.join(args.plot_dir, f"exp1_{stage}.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"\nsaved: {csv_path}")
    print(f"saved: {json_path}")
    if tracked_path:
        print(f"saved: {tracked_path}")
    print(f"saved: {plot_path}")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
