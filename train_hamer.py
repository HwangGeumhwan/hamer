"""
HaMeR 피처 기반 수화 지문자 MLP 분류기 학습

데이터 : _DATA/data/hamer_features/train_data.npz  (x: N×35, y: N)
모델   : _DATA/data/hamer_features/model/gesture_mlp.pth
그래프 : _DATA/data/hamer_features/model/train_curve.png

입력 피처 (35-dim):
  26-dim : 손 모양 (각도 11 + 손가락 끝 방향벡터 5×3)
   9-dim : MANO global_orient (손의 카메라 공간 절대 방향, 3×3 flatten)

아키텍처:
  Linear(35→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→31)
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib



ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
]
NUM_CLASSES = len(ACTIONS)


# ── Dataset ──────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)           # (N, 17)
        self.y = torch.from_numpy(y).long()    # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class GestureMLP(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dim: int = 64,
                 fc_dim: int = 32, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # (B, num_classes)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='_DATA/data/hamer_features/train_data.npz')
    parser.add_argument('--eval_data',  default='_DATA/data/hamer_features/eval_data.npz')
    parser.add_argument('--out_dir',    default='_DATA/data/hamer_features/model')
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--all',        action='store_true',
                        help='train + eval 전체를 학습에 사용 (val 없음)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    raw = np.load(args.data)
    x, y = raw['x'].astype(np.float32), raw['y'].astype(np.int64)
    print(f"Train 데이터: x={x.shape}, y={y.shape}, 클래스={NUM_CLASSES}")

    eval_raw = np.load(args.eval_data)
    ex, ey = eval_raw['x'].astype(np.float32), eval_raw['y'].astype(np.int64)
    print(f"Eval  데이터: x={ex.shape}, y={ey.shape}")

    if args.all:
        x = np.concatenate([x, ex], axis=0)
        y = np.concatenate([y, ey], axis=0)
        print(f"--all: train+eval 합산 → x={x.shape}")

    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    if not args.all:
        ex = scaler.transform(ex).astype(np.float32)
    joblib.dump(scaler, out_dir / 'scaler.pkl')
    print(f"Scaler 저장: {out_dir / 'scaler.pkl'}")

    train_ds     = GestureDataset(x, y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if not args.all:
        val_ds     = GestureDataset(ex, ey)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 모델
    model     = GestureMLP(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)

        if args.all:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train  loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        else:
            vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train  loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
                  f"Val  loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

    torch.save(model.state_dict(), out_dir / 'gesture_mlp.pth')
    print(f"\n마지막 epoch 모델 저장  →  {out_dir/'gesture_mlp.pth'}")

    # 학습 곡선 저장
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='train')
    if not args.all:
        ax1.plot(history['val_loss'], label='val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(history['train_acc'], label='train')
    if not args.all:
        ax2.plot(history['val_acc'], label='val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'train_curve.png', dpi=120)
    print(f"학습 곡선 저장: {out_dir/'train_curve.png'}")


if __name__ == '__main__':
    main()
