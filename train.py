from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from spikingjelly.activation_based import encoding, surrogate
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This script requires SpikingJelly. Install it with `pip install spikingjelly`."
    ) from e


EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def guess_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_numeric_csv(path: str) -> np.ndarray:
    p = Path(path)
    try:
        table = np.loadtxt(p, delimiter=",", dtype=np.float32)
    except Exception:
        table = np.loadtxt(p, delimiter=",", dtype=np.float32, skiprows=1)

    table = np.asarray(table, dtype=np.float32)
    if table.ndim == 1:
        table = table[None, :]
    if table.ndim != 2:
        raise ValueError("CSV must be a 2D numeric table.")
    if np.isnan(table).any():
        raise ValueError("CSV contains NaN values. Check the header or malformed rows.")
    return table


def parse_fixed_iq_csv(path: str, raw_sample_len: int = 400) -> tuple[np.ndarray, np.ndarray]:
    table = load_numeric_csv(path)
    expected_cols = 2 * raw_sample_len + 1
    if table.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns for I_1,Q_1,...,I_{raw_sample_len},Q_{raw_sample_len},label "
            f"but got {table.shape[1]}."
        )

    body = table[:, :-1]
    y = table[:, -1].astype(np.int64)

    unique = np.unique(y)
    if not np.all(np.isin(unique, np.array([0, 1]))):
        raise ValueError(f"Labels must be 0/1. Got unique labels: {unique.tolist()}")

    real = body[:, 0::2]
    imag = body[:, 1::2]
    x = np.stack([real, imag], axis=1).astype(np.float32)
    return x, y.astype(np.float32)


def split_train_val(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        if len(idx) > 1:
            n_val = min(max(n_val, 1), len(idx) - 1)
        else:
            n_val = 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


def preprocess_windows_abs(
    x_raw: torch.Tensor,
    raw_sample_len: int = 400,
    window_len: int = 320,
    window_stride: int = 1,
) -> torch.Tensor:
    if x_raw.ndim != 3 or x_raw.shape[1] != 2 or x_raw.shape[2] != raw_sample_len:
        raise ValueError(f"Expected raw IQ tensor with shape (N, 2, {raw_sample_len}).")
    if window_len != 320:
        raise ValueError("This simplified preprocessing expects 320-sample windows.")
    if raw_sample_len < window_len:
        raise ValueError("raw_sample_len must be >= window_len.")
    if window_stride <= 0:
        raise ValueError("window_stride must be positive.")

    iq = torch.complex(x_raw[:, 0, :].float(), x_raw[:, 1, :].float())
    windows = iq.unfold(dimension=1, size=window_len, step=window_stride)  # [B, W, 320]

    sts = windows[..., :160]
    lts = windows[..., 160:320]

    sts_groups = sts.reshape(sts.shape[0], sts.shape[1], 10, 16)
    kept_sts = sts_groups[:, :, [1, 2, 3, 5, 6, 7, 8, 9], :].reshape(sts.shape[0], sts.shape[1], 128)

    kept_lts = lts[..., 32:160]  # last two 64-sample groups = 128 samples

    feats = torch.cat([kept_sts, kept_lts], dim=-1)  # [B, W, 256] complex
    feats = torch.abs(feats).float()  # [B, W, 256]

    denom = feats.amax(dim=-1, keepdim=True).clamp_min(EPS)
    feats = torch.clamp(feats / denom, 0.0, 1.0)
    return feats


class SNN(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 4, T: int = 32):
        super().__init__()
        if hidden_dim != 4:
            raise ValueError("This model is fixed to 4 excitatory / 4 inhibitory neurons.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.T = T

        self.raw_w = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.15 - 2.0)
        self.readout = nn.Linear(hidden_dim, 1)
        self.sg = surrogate.ATan(alpha=2.0)

        self.register_buffer("inh_mask", torch.ones(hidden_dim, hidden_dim) - torch.eye(hidden_dim))

        self.v_e = None
        self.v_i = None
        self.theta = None

    def reset(self) -> None:
        self.v_e = None
        self.v_i = None
        self.theta = None

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (batch_size, self.hidden_dim)
        if (
            self.v_e is None
            or self.v_e.shape != shape
            or self.v_e.device != device
            or self.v_e.dtype != dtype
        ):
            self.v_e = torch.zeros(shape, device=device, dtype=dtype)
            self.v_i = torch.zeros(shape, device=device, dtype=dtype)
            self.theta = torch.ones(shape, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        self._init_state(batch_size, x.device, x.dtype)

        encoder = encoding.PoissonEncoder()
        w = torch.sigmoid(self.raw_w)
        spike_sum = torch.zeros((batch_size, self.hidden_dim), device=x.device, dtype=x.dtype)

        for _ in range(self.T):
            s_in = encoder(x).to(x.dtype)

            i_e = F.linear(s_in, w) / float(self.input_dim)
            i_e = i_e * 12.0

            self.v_e = self.v_e + (-self.v_e + i_e) / 2.0
            s_e = self.sg(self.v_e - self.theta)
            self.v_e = self.v_e * (1.0 - s_e.detach())
            self.theta = self.theta + (1.0 - self.theta) / 50.0 + 0.05 * s_e

            self.v_i = self.v_i + (-self.v_i + s_e) / 1.0
            s_i = self.sg(self.v_i - 0.5)
            self.v_i = self.v_i * (1.0 - s_i.detach())

            self.v_e = self.v_e - F.linear(s_i, self.inh_mask)

            spike_sum = spike_sum + s_e

        rate = spike_sum / float(self.T)
        return self.readout(rate).squeeze(-1)


def sample_logits_from_windows(
    model: SNN,
    x_raw: torch.Tensor,
    raw_sample_len: int,
    window_len: int,
    window_stride: int,
) -> torch.Tensor:
    feats = preprocess_windows_abs(
        x_raw,
        raw_sample_len=raw_sample_len,
        window_len=window_len,
        window_stride=window_stride,
    )  # [B, W, 256]

    b, w, d = feats.shape
    window_logits = model(feats.reshape(b * w, d)).reshape(b, w)
    sample_logits = window_logits.max(dim=1).values
    return sample_logits


def run_epoch(
    model: SNN,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    raw_sample_len: int,
    window_len: int,
    window_stride: int,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x_raw, y in loader:
        x_raw = x_raw.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        sample_logits = sample_logits_from_windows(
            model, x_raw, raw_sample_len, window_len, window_stride
        )

        loss = criterion(sample_logits, y)

        if train:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(sample_logits) >= 0.5).float()
        total_correct += int((preds == y).sum().item())
        total += int(y.numel())
        total_loss += float(loss.item()) * int(y.numel())

        model.reset()

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc


def save_model(
    path: str,
    model: SNN,
    raw_sample_len: int,
    window_len: int,
    window_stride: int,
    T: int,
    epoch: int,
    val_acc: float,
    val_loss: float,
    kind: str,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "raw_sample_len": raw_sample_len,
        "window_len": window_len,
        "window_stride": window_stride,
        "T": T,
        "threshold": 0.5,
        "epoch": epoch,
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "kind": kind,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def build_argparser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_last_model_path = str(script_dir / "last_model.pth")
    default_best_model_path = str(script_dir / "best_model.pth")
    default_train_csv = str(script_dir / "wifi_preamble.csv")
    default_valid_csv = str(script_dir / "wifi_preamble_inf.csv")

    p = argparse.ArgumentParser(description="Training script for Wi-Fi preamble detection.")
    p.add_argument("--train-data", default=default_train_csv, help="Training CSV path")
    p.add_argument("--valid-data", default=default_valid_csv, help="Validation CSV path")
    p.add_argument("--model-path", default=default_last_model_path, help="last_model.pth")
    p.add_argument("--best-model-path", default=default_best_model_path, help="best_model.pth")
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0 ...")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--raw-sample-len", type=int, default=400)
    p.add_argument("--window-len", type=int, default=320)
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--T", type=int, default=32)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = guess_device(args.device)

    x_all, y_all = parse_fixed_iq_csv(args.train_data, raw_sample_len=args.raw_sample_len)

    if args.valid_data:
        x_train, y_train = x_all, y_all
        x_val, y_val = parse_fixed_iq_csv(args.valid_data, raw_sample_len=args.raw_sample_len)
    else:
        train_idx, val_idx = split_train_val(y_all.astype(np.int64), args.val_ratio, args.seed)
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = SNN(input_dim=256, hidden_dim=4, T=args.T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print("=" * 80)
    print(f"device={device}")
    print(f"train_samples={len(x_train)} | val_samples={len(x_val)}")
    print(f"raw_sample_len={args.raw_sample_len} | window_len={args.window_len} | window_stride={args.window_stride}")
    print("=" * 80)

    last_val_acc = 0.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            raw_sample_len=args.raw_sample_len,
            window_len=args.window_len,
            window_stride=args.window_stride,
        )

        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                criterion=criterion,
                raw_sample_len=args.raw_sample_len,
                window_len=args.window_len,
                window_stride=args.window_stride,
            )

        save_model(
            args.model_path,
            model,
            raw_sample_len=args.raw_sample_len,
            window_len=args.window_len,
            window_stride=args.window_stride,
            T=args.T,
            epoch=epoch,
            val_acc=val_acc,
            val_loss=val_loss,
            kind="last",
        )

        is_better = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        )

        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            save_model(
                args.best_model_path,
                model,
                raw_sample_len=args.raw_sample_len,
                window_len=args.window_len,
                window_stride=args.window_stride,
                T=args.T,
                epoch=epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                kind="best",
            )

        last_val_acc = val_acc

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.6f} val_acc={val_acc*100:.2f}% | "
            f"best_val_acc={best_val_acc*100:.2f}% (epoch {best_epoch})"
        )

    print("=" * 80)
    print("Training finished")
    print(f"Last model saved: {Path(args.model_path).resolve()}")
    print(f"Best model saved: {Path(args.best_model_path).resolve()}")
    print(f"Last validation accuracy: {last_val_acc*100:.2f}%")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()