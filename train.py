"""Step 1 baseline: MLP on MNIST trained with SGD.

Usage:
    python train.py
    python train.py --seed 0 --batch-size 64 --lr 0.1 --epochs 5

Every run writes one CSV to results/ with per-epoch metrics, so the later
multi-seed / batch-size experiments can just call this script repeatedly.
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    p = argparse.ArgumentParser(description="MLP + SGD on MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    """Fix every random source so the same seed reproduces the same run."""
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


def get_loaders(args, device):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)

    # DataLoader shuffling is its own random source -> give it the seed too.
    g = torch.Generator()
    g.manual_seed(args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}  args={vars(args)}")

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(args, device)

    model = build_mlp(args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"step1_seed{args.seed}_bs{args.batch_size}_lr{args.lr}"
    csv_path = os.path.join(args.out_dir, tag + ".csv")

    rows = []
    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        train_time = time.time() - t0

        train_loss = loss_sum / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        row = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "epoch_time_sec": round(train_time, 3),
        }
        rows.append(row)
        print(
            f"epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test_loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"{train_time:.2f}s"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nfinal test acc: {rows[-1]['test_acc']:.4f}")
    print(f"total time: {time.time() - total_start:.2f}s")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
