"""Parse an existing tqdm training log into the new train_loss.csv format.

Useful to retroactively plot the first 50-epoch run we already have
(`outputs/diffusion_50ep_nosigmoid_train.log`). That log predates the per-sample
logging patch, so it has no `t` column — we fill it with -1 so plot_loss.py
knows to skip plot 2.

Usage:
  python scripts/parse_old_tqdm_log.py outputs/diffusion_50ep_nosigmoid_train.log \\
      --out outputs/logs/train_loss.csv

Plot 1 (train loss vs step) will then work; plots 2 and 3 will be skipped
because per-sample t / fixed-t val data aren't available from the old log.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TQDM_RE = re.compile(r"Diffusion epoch (\d+):.*?(\d+)/(\d+).*?loss=([\d.]+)")


def parse(log_path: Path) -> list[tuple[int, int, int, float]]:
    """Return list of (global_step, epoch, batch_in_epoch, batch_loss)."""
    raw = log_path.read_bytes().decode("utf-8", errors="ignore")
    # tqdm uses \r to rewrite the progress line in place; split on it so each
    # "update" becomes its own chunk.
    chunks = raw.split("\r")

    total_per_epoch: int | None = None
    # We dedupe per (epoch, batch_in_epoch) because tqdm rewrites the same
    # line many times per actual step.
    seen: dict[tuple[int, int], float] = {}
    for c in chunks:
        m = TQDM_RE.search(c)
        if not m:
            continue
        epoch = int(m.group(1))
        batch_in_epoch = int(m.group(2))
        total = int(m.group(3))
        loss = float(m.group(4))
        total_per_epoch = total
        seen[(epoch, batch_in_epoch)] = loss

    if total_per_epoch is None:
        raise SystemExit("No tqdm progress lines found in log.")

    rows: list[tuple[int, int, int, float]] = []
    for (epoch, batch_in_epoch), loss in sorted(seen.items()):
        global_step = (epoch - 1) * total_per_epoch + batch_in_epoch
        rows.append((global_step, epoch, batch_in_epoch, loss))
    return rows


def write_csv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "t", "sample_loss", "batch_loss"])
        for step, epoch, _batch, loss in rows:
            w.writerow([step, epoch, -1, "", f"{loss:.6f}"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/logs/train_loss.csv"))
    args = p.parse_args()
    rows = parse(args.log_path)
    write_csv(rows, args.out)
    print(f"parsed {len(rows):,} unique (epoch, batch) rows → {args.out}")


if __name__ == "__main__":
    main()
