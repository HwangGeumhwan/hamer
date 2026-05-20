"""
MLP 분류기 평가 (피처 기반, HaMeR 재실행 불필요)

python eval_hamer.py \
    --npz_root  _DATA/data/hamer_features/npz \
    --png_root  _DATA/data/hamer_features/png \
    --evalset   4 \
    --model     _DATA/data/hamer_features/model/gesture_mlp.pth \
    --out_dir   result

출력 (result/):
  confusion_matrix.png  : 라벨별 혼동 행렬 히트맵
  accuracy_per_label.png: 라벨별 정확도 바차트
  failed/               : 틀린 예시 이미지 (패치|스켈레톤 + 예측/정답)
"""

from pathlib import Path
import argparse
import random
import joblib
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from train_hamer import GestureMLP
from dataset_hamer import _extract_26d, ACTIONS
from korean_name_postprocess import parse_jamo_to_korean, decompose_korean_to_jamo

NUM_CLASSES = len(ACTIONS)

FONT_PATHS = [
    'Sign_Language_Translation/Sign_Language_Translation/fonts/HMKMMAG.TTF',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]

# matplotlib 한글 폰트 설정
for _fp in FONT_PATHS:
    if Path(_fp).exists():
        matplotlib.font_manager.fontManager.addfont(_fp)
        plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=_fp).get_name()
        break


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_eval_samples(data_path: Path, include_train: bool = False,
                       train_path: Path | None = None):
    """eval_data.npz (및 선택적으로 train_data.npz)에서 샘플 로드.

    반환: list of (feat, label_idx, label_str, video_stem, frame_idx)
    """
    def _load_npz(path: Path):
        raw = np.load(path, allow_pickle=True)
        x   = raw['x'].astype(np.float32)
        y   = raw['y'].astype(np.int64)
        stems  = raw['video_stems']
        fidxs  = raw['frame_indices'].astype(int)
        return [(x[i], int(y[i]), ACTIONS[int(y[i])], str(stems[i]), int(fidxs[i]))
                for i in range(len(x))]

    samples = _load_npz(data_path)
    if include_train and train_path is not None and train_path.exists():
        samples += _load_npz(train_path)
    return samples


# ── 오답 이미지 생성 ───────────────────────────────────────────────────────────

def _pil_font(size):
    for fp in FONT_PATHS:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def make_failed_image(png_root: Path, label_str, video_stem, frame_idx,
                      top3: list[tuple[str, float]]):
    """PNG(패치|스켈레톤)를 불러와 예측/정답 패널을 붙인 이미지 반환.

    Args:
        top3: [(label, conf), ...] — 확률 상위 3개 예측 (내림차순)
    """
    png_path = png_root / label_str / f'{video_stem}_f{frame_idx:04d}.png'
    if not png_path.exists():
        return None
    img = cv2.imread(str(png_path))
    if img is None:
        return None

    H = 256
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * H / h), H))

    panel = np.full((H, 210, 3), 30, dtype=np.uint8)
    pil   = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil)

    f_small = _pil_font(18)
    f_big   = _pil_font(50)
    f_mid   = _pil_font(30)

    # top-1 예측
    draw.text((10,  8),  '예측',                             font=f_small, fill=(200, 200, 200))
    draw.text((10, 28),  top3[0][0],                         font=f_big,   fill=(220, 100, 100))
    draw.text((10, 82),  f'{top3[0][1]*100:.1f}%',           font=f_small, fill=(180, 180, 180))

    # top-2, top-3 예측
    if len(top3) > 1:
        draw.text((10, 108), f'2위 {top3[1][0]}  {top3[1][1]*100:.1f}%', font=f_small, fill=(160, 160, 100))
    if len(top3) > 2:
        draw.text((10, 130), f'3위 {top3[2][0]}  {top3[2][1]*100:.1f}%', font=f_small, fill=(130, 130, 130))

    # 정답
    draw.text((10, 158), '정답',                             font=f_small, fill=(200, 200, 200))
    draw.text((10, 178), label_str,                          font=f_mid,   fill=(100, 220, 100))

    panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return np.concatenate([img, panel], axis=1)


# ── 혼동 행렬 히트맵 ──────────────────────────────────────────────────────────

def save_confusion_matrix(conf_mat: np.ndarray, out_path: Path):
    row_sum = conf_mat.sum(axis=1, keepdims=True).clip(min=1)
    norm    = conf_mat / row_sum  # 행별 정규화 → 대각선 = 클래스별 정확도

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTIONS, fontsize=11)
    ax.set_yticklabels(ACTIONS, fontsize=11)
    ax.set_xlabel('예측', fontsize=13)
    ax.set_ylabel('정답', fontsize=13)
    ax.set_title('라벨별 혼동 행렬  (행: 정답, 열: 예측 / 색상: 행 내 비율)', fontsize=13)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cnt = conf_mat[i, j]
            if cnt == 0:
                continue
            color = 'white' if norm[i, j] > 0.55 else 'black'
            ax.text(j, i, str(cnt), ha='center', va='center',
                    fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"혼동 행렬 저장: {out_path}")


# ── 라벨별 정확도 바차트 ──────────────────────────────────────────────────────

def save_accuracy_bar(conf_mat: np.ndarray, out_path: Path):
    row_sum = conf_mat.sum(axis=1).clip(min=1)
    acc     = conf_mat.diagonal() / row_sum

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(ACTIONS, acc, color=['#4C9BE8' if a >= 0.8 else '#E87C4C' for a in acc])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('정확도', fontsize=12)
    ax.set_title('라벨별 정확도', fontsize=13)
    ax.axhline(acc.mean(), color='gray', linestyle='--', linewidth=1,
               label=f'평균 {acc.mean()*100:.1f}%')
    ax.legend(fontsize=11)

    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{a*100:.0f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"정확도 바차트 저장: {out_path}")


# ── 복원 모드 공통 유틸 ───────────────────────────────────────────────────────

def _load_model(model_path: str, device: torch.device):
    ckpt      = torch.load(model_path, map_location=device)
    input_dim = ckpt['net.0.weight'].shape[1]
    model     = GestureMLP(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    scaler_path = Path(model_path).parent / 'scaler.pkl'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is None:
        print('[WARN] scaler.pkl 없음 — 정규화 미적용')
    return model, scaler


def _apply_scaler(feat: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return feat
    return scaler.transform(feat[None])[0].astype(np.float32)


def _predict_feat(feat: np.ndarray, model, device, scaler=None) -> tuple[str, float]:
    feat = _apply_scaler(feat, scaler)
    feat_t = torch.from_numpy(feat[None]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(feat_t), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return ACTIONS[idx], float(probs[idx])


def _load_feat_from_npz(npz_root: Path, label_str: str,
                         video_stem: str, frame_idx: int) -> np.ndarray:
    """seq.npz + frame npz에서 특정 프레임의 피처를 반환합니다."""
    from dataset_hamer import _finger_tip_dists
    npz_path  = npz_root / label_str / f'{video_stem}_seq.npz'
    data      = np.load(npz_path, allow_pickle=True)
    fidx_list = data['frame_indices'].tolist()
    pos       = fidx_list.index(frame_idx)
    feat_26d  = _extract_26d(data['features'][pos])
    parts     = [feat_26d]
    if 'global_orients' in data:
        parts.append(data['global_orients'][pos].astype(np.float32))
    frame_npz = npz_root / label_str / f'{video_stem}_f{frame_idx:04d}.npz'
    parts.append(_finger_tip_dists(frame_npz))
    return np.concatenate(parts)


def _parse_png_stem(img_path: Path) -> tuple[str, str, int]:
    """png 경로 → (label_str, video_stem, frame_idx)"""
    label_str  = img_path.parent.name
    stem       = img_path.stem          # e.g. 'video1_f0042'
    parts      = stem.rsplit('_f', 1)
    if len(parts) != 2:
        raise ValueError(f"파일명 형식 오류: {img_path.name}  (expected: {{stem}}_f{{idx:04d}}.png)")
    return label_str, parts[0], int(parts[1])


def _load_all_samples(npz_root: Path) -> dict[str, list]:
    """npz_root 전체를 읽어 {label_str: [(feat, video_stem, frame_idx), ...]} 반환."""
    from dataset_hamer import _finger_tip_dists
    by_label: dict[str, list] = {}
    for label_dir in sorted(npz_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label_str = label_dir.name
        by_label.setdefault(label_str, [])
        for seq_file in sorted(label_dir.glob('*_seq.npz')):
            video_stem = seq_file.stem.removesuffix('_seq')
            data = np.load(seq_file, allow_pickle=True)
            if 'features' not in data:
                continue
            base  = np.array([_extract_26d(f) for f in data['features']])
            parts = [base]
            if 'global_orients' in data:
                parts.append(data['global_orients'].astype(np.float32))
            feats         = np.concatenate(parts, axis=1) if len(parts) > 1 else base
            frame_indices = data['frame_indices'].tolist()
            for feat, fidx in zip(feats, frame_indices):
                frame_npz = seq_file.parent / f'{video_stem}_f{int(fidx):04d}.npz'
                dists = _finger_tip_dists(frame_npz)
                by_label[label_str].append((np.concatenate([feat, dists]), video_stem, int(fidx)))
    return by_label


def _make_restore_strip(panels: list[tuple], restored: str,
                         font_path: str) -> np.ndarray:
    """각 자모 패널을 수평 나열하고 복원 결과를 아래에 붙인 이미지를 반환합니다.

    panels: list of (img_bgr_or_None, pred_jamo, gt_jamo_or_None, conf)
    """
    cell_h, cell_w = 256, 200
    n = len(panels)

    try:
        f_big   = ImageFont.truetype(font_path, 60)
        f_small = ImageFont.truetype(font_path, 22)
        f_res   = ImageFont.truetype(font_path, 72)
    except Exception:
        f_big = f_small = f_res = ImageFont.load_default()

    cells = []
    for img_bgr, pred_jamo, gt_jamo, conf in panels:
        cell = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            scale = min((cell_h - 80) / h, (cell_w - 10) / w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img_bgr, (nw, nh))
            yo = (cell_h - 80 - nh) // 2
            xo = (cell_w - nw) // 2
            cell[yo:yo+nh, xo:xo+nw] = resized

        pil  = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        # 예측 자모
        color = (100, 220, 100) if gt_jamo is None or pred_jamo == gt_jamo else (220, 100, 100)
        draw.text((cell_w // 2 - 20, cell_h - 75), pred_jamo, font=f_big, fill=color)
        draw.text((5, cell_h - 20), f'{conf*100:.0f}%', font=f_small, fill=(180, 180, 180))
        if gt_jamo is not None and pred_jamo != gt_jamo:
            draw.text((cell_w - 45, cell_h - 20), f'({gt_jamo})', font=f_small, fill=(100, 180, 255))

        cells.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    strip = np.concatenate(cells, axis=1)

    # 복원 결과 패널
    result_panel = np.full((100, strip.shape[1], 3), 20, dtype=np.uint8)
    pil_r = Image.fromarray(cv2.cvtColor(result_panel, cv2.COLOR_BGR2RGB))
    draw_r = ImageDraw.Draw(pil_r)
    draw_r.text((10, 14), f'복원: {restored}', font=f_res, fill=(255, 220, 60))
    result_panel = cv2.cvtColor(np.array(pil_r), cv2.COLOR_RGB2BGR)

    return np.concatenate([strip, result_panel], axis=0)


# ── 복원 모드 1: 이미지 시퀀스 → 글자 복원 ────────────────────────────────────

def restore_from_images(img_paths: list[str], npz_root: Path,
                         model, device, out_dir: Path, font_path: str, scaler=None):
    """순서 있는 PNG 파일 목록에서 자모를 예측하고 한국어 이름을 복원합니다.

    각 이미지 경로는 png_root/label_str/video_stem_f{idx:04d}.png 형식이어야 합니다.
    """
    panels: list[tuple] = []
    pred_jamos: list[str] = []

    for img_path_str in img_paths:
        img_path = Path(img_path_str)
        label_str, video_stem, frame_idx = _parse_png_stem(img_path)

        feat = _load_feat_from_npz(npz_root, label_str, video_stem, frame_idx)
        pred_jamo, conf = _predict_feat(feat, model, device, scaler=scaler)
        pred_jamos.append(pred_jamo)

        img_bgr = cv2.imread(str(img_path)) if img_path.exists() else None
        panels.append((img_bgr, pred_jamo, None, conf))

    restored = parse_jamo_to_korean(pred_jamos)
    print(f"[복원] 예측 자모 : {' '.join(pred_jamos)}")
    print(f"[복원] 복원 결과 : {restored}")

    out_dir.mkdir(parents=True, exist_ok=True)
    strip = _make_restore_strip(panels, restored, font_path)
    out_path = out_dir / 'restored_from_images.png'
    cv2.imwrite(str(out_path), strip)
    print(f"[복원] 결과 이미지: {out_path}")

    return pred_jamos, restored


# ── 복원 모드 2: 글자 입력 → 테스트셋 무작위 추출 → 글자 복원 ────────────────

def restore_from_text(text: str, npz_root: Path, png_root: Path | None,
                       model, device, out_dir: Path, font_path: str, seed: int | None = None,
                       scaler=None):
    """입력 이름을 자모로 분해하고 각 자모에 대해 데이터셋에서 무작위 샘플을 추출하여
    예측 후 이름을 복원합니다.

    Args:
        text: 복원할 한국어 이름 (예: '홍길동')
        seed: 재현성을 위한 랜덤 시드 (None이면 랜덤)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    jamo_seq = decompose_korean_to_jamo(text)
    print(f"[복원] 입력      : {text}")
    print(f"[복원] 자모 분해 : {' '.join(jamo_seq)}")

    by_label = _load_all_samples(npz_root)

    panels: list[tuple] = []
    pred_jamos: list[str] = []

    for gt_jamo in jamo_seq:
        samples = by_label.get(gt_jamo, [])
        if not samples:
            print(f"[WARN] '{gt_jamo}' 샘플 없음 — 원본 자모로 대체")
            pred_jamos.append(gt_jamo)
            panels.append((None, gt_jamo, gt_jamo, 1.0))
            continue

        feat, video_stem, frame_idx = random.choice(samples)
        pred_jamo, conf = _predict_feat(feat, model, device, scaler=scaler)
        pred_jamos.append(pred_jamo)

        img_bgr = None
        if png_root is not None:
            png_path = png_root / gt_jamo / f'{video_stem}_f{frame_idx:04d}.png'
            if png_path.exists():
                img_bgr = cv2.imread(str(png_path))

        panels.append((img_bgr, pred_jamo, gt_jamo, conf))

    restored = parse_jamo_to_korean(pred_jamos)
    match = '✓' if restored == text else '✗'
    print(f"[복원] 예측 자모 : {' '.join(pred_jamos)}")
    print(f"[복원] 복원 결과 : {restored}  {match}  (원본: {text})")

    out_dir.mkdir(parents=True, exist_ok=True)
    strip = _make_restore_strip(panels, f"{restored}  ({text})", font_path)
    safe_text = text.replace('/', '_')
    out_path  = out_dir / f'restored_{safe_text}.png'
    cv2.imwrite(str(out_path), strip)
    print(f"[복원] 결과 이미지: {out_path}")

    return jamo_seq, pred_jamos, restored


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HaMeR 지문자 분류기 평가 + 한국어 이름 복원')
    parser.add_argument('--npz_root',   default='_DATA/data/hamer_features/npz')
    parser.add_argument('--png_root',   default='_DATA/data/hamer_features/png',
                        help='오답/복원 이미지 원본 폴더 (없으면 이미지 패널 생략)')
    parser.add_argument('--eval_data',  default='_DATA/data/hamer_features/eval_data.npz')
    parser.add_argument('--train_data', default='_DATA/data/hamer_features/train_data.npz')
    parser.add_argument('--all',        action='store_true',
                        help='train_data.npz까지 포함해 전체 데이터로 평가')
    parser.add_argument('--model',    default='_DATA/data/hamer_features/model/gesture_mlp.pth')
    parser.add_argument('--out_dir',  default='result')
    # ── 복원 모드 ──────────────────────────────────────────────────────────────
    parser.add_argument('--restore_images', nargs='+', metavar='PNG',
                        help='이미지 시퀀스(PNG 경로 목록)에서 한국어 이름을 복원합니다.')
    parser.add_argument('--restore_text', metavar='TEXT',
                        help='한국어 이름을 입력하면 데이터셋에서 자모별 무작위 샘플을 추출하여 복원합니다.')
    parser.add_argument('--seed', type=int, default=None,
                        help='--restore_text 모드의 랜덤 시드 (재현성)')
    args = parser.parse_args()

    # ── 복원 모드 진입 ─────────────────────────────────────────────────────────
    if args.restore_images or args.restore_text:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, scaler = _load_model(args.model, device)
        npz_root  = Path(args.npz_root)
        png_root  = Path(args.png_root) if Path(args.png_root).exists() else None
        out_dir   = Path(args.out_dir)
        font_path = next((fp for fp in FONT_PATHS if Path(fp).exists()), '')

        if args.restore_images:
            restore_from_images(args.restore_images, npz_root,
                                 model, device, out_dir, font_path, scaler=scaler)
        if args.restore_text:
            restore_from_text(args.restore_text, npz_root, png_root,
                               model, device, out_dir, font_path, seed=args.seed,
                               scaler=scaler)
        return

    out_dir    = Path(args.out_dir)
    failed_dir = out_dir / 'failed'
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    png_root = Path(args.png_root)
    use_png  = png_root.exists()
    if not use_png:
        print(f"[INFO] png_root 없음 — 오답 이미지 저장 생략: {png_root}")

    # ── 샘플 로드 ──────────────────────────────────────────────────────────────
    samples = load_eval_samples(Path(args.eval_data), include_train=args.all,
                                train_path=Path(args.train_data))
    scope   = 'train+eval 전체' if args.all else 'eval'
    print(f"Eval 샘플 수: {len(samples)}  ({scope})")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scaler = _load_model(args.model, device)
    input_dim = model.net[0].in_features
    print(f"모델 로드: {args.model}  (input_dim={input_dim})")

    feat_dim = samples[0][0].shape[0] if samples else input_dim
    if feat_dim != input_dim:
        raise RuntimeError(
            f"피처 차원({feat_dim})이 모델 입력 차원({input_dim})과 다릅니다. "
            f"학습 시 사용한 피처 구성(global_orients 포함 여부)을 확인하세요."
        )

    # ── 추론 ───────────────────────────────────────────────────────────────────
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    correct  = 0

    for feat, gt_idx, label_str, video_stem, frame_idx in samples:
        feat_scaled = _apply_scaler(feat, scaler)
        feat_t = torch.from_numpy(feat_scaled[None]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(feat_t), dim=1)[0].cpu().numpy()

        top3_idx   = probs.argsort()[::-1][:3]
        top3       = [(ACTIONS[i], float(probs[i])) for i in top3_idx]
        pred_idx   = int(top3_idx[0])
        pred_label = top3[0][0]
        conf_mat[gt_idx, pred_idx] += 1

        if pred_idx == gt_idx:
            correct += 1
        elif use_png:
            img = make_failed_image(png_root, label_str, video_stem,
                                    frame_idx, top3)
            if img is not None:
                fname = f'{label_str}_pred{pred_label}_{video_stem}_f{frame_idx:04d}.png'
                cv2.imwrite(str(failed_dir / fname), img)

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    total = len(samples)
    print(f"\n전체 정확도: {correct}/{total} = {correct/total*100:.1f}%")
    print("\n라벨별 정확도:")
    for i, action in enumerate(ACTIONS):
        row = conf_mat[i]
        n   = row.sum()
        if n == 0:
            continue
        acc = conf_mat[i, i] / n
        wrongs = [(ACTIONS[j], row[j]) for j in range(NUM_CLASSES)
                  if j != i and row[j] > 0]
        wrongs_str = ', '.join(f'{a}({c})' for a, c in
                               sorted(wrongs, key=lambda x: -x[1]))
        print(f"  {action}: {acc*100:.1f}%  오답→ {wrongs_str or '없음'}")

    # ── 플롯 저장 ──────────────────────────────────────────────────────────────
    save_confusion_matrix(conf_mat, out_dir / 'confusion_matrix.png')
    save_accuracy_bar(conf_mat,     out_dir / 'accuracy_per_label.png')
    print(f"\n오답 이미지: {failed_dir}/")


if __name__ == '__main__':
    main()
