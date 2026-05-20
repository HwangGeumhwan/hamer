"""
데이터셋 생성: _seq.npz → train_data.npz

*_seq.npz (features_26d: N×26, label_idx) 파일을 읽어
각 프레임을 독립 샘플로 수집하고
_DATA/data/hamer_features/train_data.npz 로 저장합니다.

  x : (N, 35)   26 (손 모양) + 9 (global_orient)
  y : (N,)      라벨 인덱스 (0~30)
"""

from pathlib import Path
import numpy as np
import argparse


def _extract_from_75d(feat_75: np.ndarray) -> np.ndarray:
    """75-dim 피처 (vectors 60-dim + angles 15-dim) → 26-dim 증류 피처.

    구성 (총 26-dim):
      angles[0:3]     엄지 전체 각도               (3-dim)
      angles[3:5]     검지 MCP·PIP 각도            (2-dim)
      angles[6:8]     중지 MCP·PIP 각도            (2-dim)
      angles[9:11]    약지 MCP·PIP 각도            (2-dim)
      angles[12:14]   소지 MCP·PIP 각도            (2-dim)
      vectors[3]      엄지 DIP→TIP 방향벡터        (3-dim)
      vectors[7]      검지 DIP→TIP 방향벡터        (3-dim)
      vectors[11]     중지 DIP→TIP 방향벡터        (3-dim)
      vectors[15]     약지 DIP→TIP 방향벡터        (3-dim)
      vectors[19]     소지 DIP→TIP 방향벡터        (3-dim)
    """
    vectors = feat_75[:60].reshape(20, 3)
    angles  = feat_75[60:]
    return np.concatenate([
        angles[0:3],
        angles[3:5],
        angles[6:8],
        angles[9:11],
        angles[12:14],
        vectors[3],
        vectors[7],
        vectors[11],
        vectors[15],
        vectors[19],
    ]).astype(np.float32)

_extract_26d = _extract_from_75d


def _finger_tip_dists(frame_npz_path: Path) -> np.ndarray:
    """keypoints_3d (21,3) → 정규화된 검-중, 중-약 손끝 거리 (2-dim).

    정규화 기준: 손목(0) - 중지 MCP(9) 거리.
    frame npz가 없거나 keypoints_3d 키가 없으면 zeros 반환.
    """
    if not frame_npz_path.exists():
        return np.zeros(2, dtype=np.float32)
    d = np.load(frame_npz_path)
    if 'keypoints_3d' not in d:
        return np.zeros(2, dtype=np.float32)
    kp    = d['keypoints_3d']                          # (21, 3)
    scale = np.linalg.norm(kp[9] - kp[12]) + 1e-8     # 중지 MCP - 중지 TIP (손가락 길이)
    return np.array([
        np.linalg.norm(kp[8]  - kp[12]) / scale,      # 검지 TIP - 중지 TIP
        np.linalg.norm(kp[12] - kp[16]) / scale,      # 중지 TIP - 약지 TIP
    ], dtype=np.float32)


ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
]


def build_dataset(npz_root: Path, evalset: int | None = None):
    train_xs, train_ys, train_stems, train_fidxs = [], [], [], []
    eval_xs,  eval_ys,  eval_stems,  eval_fidxs  = [], [], [], []
    missing = []

    for label_dir in sorted(npz_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label_str = label_dir.name
        seq_files = sorted(label_dir.glob('*_seq.npz'))
        if not seq_files:
            missing.append(label_str)
            continue

        for seq_file in seq_files:
            data = np.load(seq_file, allow_pickle=True)
            if 'features' not in data:
                print(f"  [SKIP] features 없음: {seq_file.name}")
                continue

            base  = np.array([_extract_from_75d(f) for f in data['features']])  # (N, 26)
            parts = [base]
            if 'global_orients' in data:
                parts.append(data['global_orients'].astype(np.float32))    # +9
            feats = np.concatenate(parts, axis=1) if len(parts) > 1 else base
            label_idx = int(data['label_idx'].flat[0])
            video_stem = seq_file.stem.removesuffix('_seq')

            is_eval = False
            if evalset is not None:
                try:
                    is_eval = int(video_stem.rsplit('_', 1)[-1]) == evalset
                except ValueError:
                    pass

            xs, ys, stms, fids = (
                (eval_xs, eval_ys, eval_stems, eval_fidxs) if is_eval
                else (train_xs, train_ys, train_stems, train_fidxs)
            )
            for feat, fidx in zip(feats, data['frame_indices']):
                frame_npz = seq_file.parent / f'{video_stem}_f{int(fidx):04d}.npz'
                dists = _finger_tip_dists(frame_npz)
                xs.append(np.concatenate([feat, dists]))
                ys.append(label_idx)
                stms.append(video_stem)
                fids.append(int(fidx))

    if missing:
        print(f"[WARN] seq.npz 없는 라벨: {missing}")

    train = (np.stack(train_xs), np.array(train_ys, dtype=np.int64),
             np.array(train_stems), np.array(train_fidxs, dtype=np.int64))
    eval_ = (np.stack(eval_xs), np.array(eval_ys, dtype=np.int64),
             np.array(eval_stems), np.array(eval_fidxs, dtype=np.int64)) if eval_xs else None
    return train, eval_


def _print_stats(label: str, x: np.ndarray, y: np.ndarray):
    print(f"\n{label}: x={x.shape}, y={y.shape}")
    for idx, action in enumerate(ACTIONS):
        count = (y == idx).sum()
        if count > 0:
            print(f"    [{idx:2d}] {action} : {count}개")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_root', default='_DATA/data/hamer_features/npz')
    parser.add_argument('--out',      default='_DATA/data/hamer_features/train_data.npz')
    parser.add_argument('--evalset',  type=int, default=2,
                        help='eval set으로 쓸 데이터 번호 (예: 2 → {label}_2_seq.npz를 eval로 분리)')
    parser.add_argument('--eval_out', default='_DATA/data/hamer_features/eval_data.npz')
    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"NPZ 루트: {npz_root}")
    if args.evalset is not None:
        print(f"Eval set: 데이터 번호 {args.evalset}")

    (train_x, train_y, train_stems, train_fidxs), eval_split = build_dataset(npz_root, args.evalset)

    np.savez(out_path, x=train_x, y=train_y,
             video_stems=train_stems, frame_indices=train_fidxs)
    print(f"\n저장 완료: {out_path}")
    _print_stats("Train", train_x, train_y)

    if eval_split is not None:
        eval_x, eval_y, eval_stems, eval_fidxs = eval_split
        eval_path = Path(args.eval_out)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(eval_path, x=eval_x, y=eval_y,
                 video_stems=eval_stems, frame_indices=eval_fidxs)
        print(f"\n저장 완료: {eval_path}")
        _print_stats("Eval", eval_x, eval_y)


if __name__ == '__main__':
    main()
