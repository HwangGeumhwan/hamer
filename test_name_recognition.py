"""
이름 인식 end-to-end 테스트 (단계별 시간 측정 + 로그)

dataset/name_test/{name}/{0..N}.jpg 이미지를 순서대로 읽어
HaMeR → 피처 추출 → MLP 분류 → 한글 조합 파이프라인을 실행합니다.

  - 모델 로딩부터 각 이미지 처리까지 단계별 시간 측정
  - 결과를 콘솔 출력과 로그 파일에 동시 기록
  - 오답(failure case) 시각화 이미지 저장

사용:
  python CODE/test_name_recognition.py
  python CODE/test_name_recognition.py --names 김연아 손흥민
  python CODE/test_name_recognition.py --out_dir result/name_test --save_all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import argparse
import time
import datetime
import cv2
import numpy as np
import torch
import joblib
from PIL import Image, ImageDraw, ImageFont

from extract_hamer_features import detect_hands, _vector_normalization_3d, ACTIONS
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from vitpose_model import ViTPoseModel
from train_hamer import GestureMLP
from korean_name_postprocess import parse_jamo_to_korean, decompose_korean_to_jamo


FONT_PATHS = [
    'Sign_Language_Translation/Sign_Language_Translation/fonts/HMKMMAG.TTF',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]

_LOG_FILE = None   # main()에서 설정


# ── 로그 ───────────────────────────────────────────────────────────────────────

def log(msg: str = ''):
    print(msg)
    if _LOG_FILE is not None:
        print(msg, file=_LOG_FILE, flush=True)


# ── 타이머 헬퍼 ────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._t = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t

    def reset(self) -> float:
        e = self.elapsed()
        self._t = time.perf_counter()
        return e


# ── 피처 추출 ──────────────────────────────────────────────────────────────────

def _extract_26d(feat_75: np.ndarray) -> np.ndarray:
    vectors = feat_75[:60].reshape(20, 3)
    angles  = feat_75[60:]
    return np.concatenate([
        angles[0:3], angles[3:5], angles[6:8], angles[9:11], angles[12:14],
        vectors[3], vectors[7], vectors[11], vectors[15], vectors[19],
    ]).astype(np.float32)


def _fingertip_dists(kp3d: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(kp3d[9] - kp3d[12]) + 1e-8
    return np.array([
        np.linalg.norm(kp3d[8]  - kp3d[12]) / scale,
        np.linalg.norm(kp3d[12] - kp3d[16]) / scale,
    ], dtype=np.float32)


def extract_features(img_bgr, hamer_model, model_cfg, detector, cpm,
                     device, rescale_factor=2.0):
    """이미지 한 장 → 37-dim 피처. 손 미검출 시 None 반환."""
    bboxes, sides, _, _, _ = detect_hands(img_bgr, detector, cpm)
    if bboxes is None:
        return None
    right_mask = sides == 1
    if not right_mask.any():
        return None

    r_bboxes = bboxes[right_mask][:1]
    r_sides  = sides[right_mask][:1]
    dataset  = ViTDetDataset(model_cfg, img_bgr, r_bboxes, r_sides,
                              rescale_factor=rescale_factor)
    batch = recursive_to(next(iter(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    )), device)

    with torch.no_grad():
        out = hamer_model(batch)

    kp3d          = out['pred_keypoints_3d'][0].detach().cpu().numpy()
    global_orient = (out['pred_mano_params']['global_orient'][0]
                     .detach().cpu().numpy().reshape(9).astype(np.float32))
    vectors, angles = _vector_normalization_3d(kp3d)
    feat75 = np.concatenate([vectors.flatten(), angles.flatten()])

    del out, batch
    torch.cuda.empty_cache()

    return np.concatenate([_extract_26d(feat75), global_orient,
                            _fingertip_dists(kp3d)]).astype(np.float32)


# ── MLP 분류 ───────────────────────────────────────────────────────────────────

def load_classifier(model_path: str, device: torch.device):
    ckpt      = torch.load(model_path, map_location=device)
    input_dim = ckpt['net.0.weight'].shape[1]
    mlp       = GestureMLP(input_dim=input_dim).to(device)
    mlp.load_state_dict(ckpt)
    mlp.eval()
    scaler_path = Path(model_path).parent / 'scaler.pkl'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return mlp, scaler


def classify(feat: np.ndarray, mlp, device, scaler=None):
    if scaler is not None:
        feat = scaler.transform(feat[None])[0].astype(np.float32)
    with torch.no_grad():
        probs = torch.softmax(
            mlp(torch.from_numpy(feat[None]).to(device)), dim=1
        )[0].cpu().numpy()
    idx = int(probs.argmax())
    return ACTIONS[idx], float(probs[idx])


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def _get_font(size: int):
    for fp in FONT_PATHS:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_result_strip(panels, gt_name: str, pred_name: str,
                       per_img_times: list[float]) -> np.ndarray:
    """각 자모 셀(입력 이미지 + 예측 자모 + 시간)을 가로로 나열하고
    결과 패널을 아래에 붙인 이미지를 반환합니다.

    panels: list of (img_bgr | None, pred_jamo, gt_jamo, conf)
    """
    CELL_H, CELL_W = 300, 185
    IMG_H          = CELL_H - 100

    f_jamo  = _get_font(52)
    f_small = _get_font(18)
    f_res   = _get_font(56)

    cells = []
    for (img_bgr, pred_jamo, gt_jamo, conf), t_img in zip(panels, per_img_times):
        cell = np.full((CELL_H, CELL_W, 3), 28, dtype=np.uint8)
        if img_bgr is not None:
            h, w   = img_bgr.shape[:2]
            scale  = min(IMG_H / h, CELL_W / w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img_bgr, (nw, nh))
            yo = (IMG_H - nh) // 2
            xo = (CELL_W - nw) // 2
            cell[yo:yo + nh, xo:xo + nw] = resized

        pil  = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        correct    = (pred_jamo == gt_jamo)
        jamo_color = (100, 220, 100) if correct else (220, 80, 80)

        draw.text((CELL_W // 2 - 22, CELL_H - 95), pred_jamo,
                  font=f_jamo, fill=jamo_color)
        if not correct:
            draw.text((5, CELL_H - 38), f'({gt_jamo})',
                      font=f_small, fill=(100, 160, 255))
        draw.text((CELL_W - 65, CELL_H - 38), f'{conf*100:.0f}%',
                  font=f_small, fill=(160, 160, 160))
        draw.text((5, CELL_H - 20), f'{t_img*1000:.0f}ms',
                  font=f_small, fill=(120, 120, 120))

        cells.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    strip = np.concatenate(cells, axis=1)

    match     = pred_name == gt_name.replace(' ', '')
    bg_color  = (20, 42, 20) if match else (42, 20, 20)
    res_panel = np.full((88, strip.shape[1], 3), bg_color, dtype=np.uint8)
    pil_r  = Image.fromarray(cv2.cvtColor(res_panel, cv2.COLOR_BGR2RGB))
    draw_r = ImageDraw.Draw(pil_r)
    clr    = (80, 220, 80) if match else (220, 80, 80)
    sym    = '✓' if match else '✗'
    draw_r.text((12, 10), f'{sym}  예측: {pred_name}    정답: {gt_name}',
                font=f_res, fill=clr)
    res_panel = cv2.cvtColor(np.array(pil_r), cv2.COLOR_RGB2BGR)

    return np.concatenate([strip, res_panel], axis=0)


# ── 이름 테스트 ────────────────────────────────────────────────────────────────

def test_one_name(name_dir: Path, gt_jamos: list[str],
                  mlp, device, hamer_model, model_cfg,
                  detector, cpm, scaler, rescale_factor=2.0):
    """이미지 시퀀스 처리. 이미지별 시간을 측정합니다."""
    imgs = sorted(name_dir.glob('*.jpg'), key=lambda p: int(p.stem))

    panels, pred_jamos, per_img_times = [], [], []
    name_timer = Timer()

    for i, img_path in enumerate(imgs):
        img_timer = Timer()
        img = cv2.imread(str(img_path))
        gt_jamo = gt_jamos[i] if i < len(gt_jamos) else '?'

        feat = (extract_features(img, hamer_model, model_cfg, detector,
                                  cpm, device, rescale_factor)
                if img is not None else None)

        if feat is None:
            pred_jamo, conf = gt_jamo, 0.0
            img = None
        else:
            pred_jamo, conf = classify(feat, mlp, device, scaler)

        t_img = img_timer.elapsed()
        pred_jamos.append(pred_jamo)
        per_img_times.append(t_img)
        panels.append((img, pred_jamo, gt_jamo, conf))

        mark = '✓' if pred_jamo == gt_jamo else '✗'
        log(f"    [{i:02d}] {img_path.name:<8}  GT={gt_jamo}  Pred={pred_jamo} "
            f"{mark}  conf={conf*100:.1f}%  {t_img*1000:.0f}ms")

    elapsed  = name_timer.elapsed()
    restored = parse_jamo_to_korean(pred_jamos)
    return panels, pred_jamos, restored, elapsed, per_img_times


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    global _LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_test',      default='dataset/name_test')
    parser.add_argument('--model',          default='_DATA/data/hamer_features/model/gesture_mlp.pth')
    parser.add_argument('--checkpoint',     default=DEFAULT_CHECKPOINT)
    parser.add_argument('--body_detector',  default='regnety', choices=['vitdet', 'regnety'])
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--out_dir',        default='result/name_test')
    parser.add_argument('--save_all',       action='store_true',
                        help='오답뿐 아니라 전체 결과 이미지도 저장')
    parser.add_argument('--names',          nargs='+', default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = out_dir / f'test_{ts}.log'
    _LOG_FILE = open(log_path, 'w', encoding='utf-8')

    total_timer = Timer()

    log(f"{'='*70}")
    log(f"이름 인식 end-to-end 테스트  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*70}")

    # ── 모델 로딩 ──────────────────────────────────────────────────────────────
    log('\n[1] 모델 로딩')

    t = Timer()
    download_models(CACHE_DIR_HAMER)
    log(f'  HaMeR 체크포인트 확인      {t.reset():.2f}s')

    hamer_model, model_cfg = load_hamer(args.checkpoint, init_renderer=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hamer_model = hamer_model.to(device).eval()
    log(f'  HaMeR 모델 로드           {t.reset():.2f}s  (device={device})')

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer as hamer_pkg
        cfg_path = Path(hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        det_cfg  = LazyConfig.load(str(cfg_path))
        det_cfg.train.init_checkpoint = (
            'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
            'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl')
        for i in range(3):
            det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(det_cfg)
    else:
        from detectron2 import model_zoo
        det_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        det_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        det_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(det_cfg)
    log(f'  Detectron2({args.body_detector}) 로드  {t.reset():.2f}s')

    cpm = ViTPoseModel(device)
    log(f'  ViTPose 로드              {t.reset():.2f}s')

    mlp, scaler = load_classifier(args.model, device)
    log(f'  MLP 분류기 로드           {t.reset():.2f}s  '
        f'(input_dim={mlp.net[0].in_features}, scaler={"있음" if scaler else "없음"})')

    # ── 테스트 대상 ────────────────────────────────────────────────────────────
    name_test = Path(args.name_test)
    name_dirs = sorted([d for d in name_test.iterdir() if d.is_dir()])
    if args.names:
        name_dirs = [d for d in name_dirs if d.name in args.names]

    log(f'\n[2] 테스트 시작  ({len(name_dirs)}개 이름)')
    log('-' * 70)

    total_names   = len(name_dirs)
    correct_count = 0
    total_infer   = 0.0
    failures      = []

    for n_idx, name_dir in enumerate(name_dirs, 1):
        gt_name    = name_dir.name
        gt_nospace = gt_name.replace(' ', '')
        gt_jamos   = decompose_korean_to_jamo(gt_nospace)

        log(f'\n[{n_idx:02d}/{total_names}] {gt_name}  ({len(gt_jamos)}자모)')
        log(f'  GT 자모: {" ".join(gt_jamos)}')

        panels, pred_jamos, restored, elapsed, per_img_times = test_one_name(
            name_dir, gt_jamos, mlp, device, hamer_model, model_cfg,
            detector, cpm, scaler, args.rescale_factor)

        total_infer += elapsed
        match = (restored == gt_nospace)
        if match:
            correct_count += 1

        status = '✓' if match else '✗'
        avg_ms = elapsed / len(gt_jamos) * 1000
        log(f'  → 예측: {restored}  {status}  '
            f'총 {elapsed:.2f}s  (이미지당 평균 {avg_ms:.0f}ms)')

        if not match:
            failures.append((gt_name, gt_jamos, pred_jamos, restored, per_img_times))

        save = not match or args.save_all
        if save:
            strip = make_result_strip(panels, gt_name, restored, per_img_times)
            prefix = 'OK' if match else 'FAIL'
            safe   = gt_name.replace(' ', '_')
            cv2.imwrite(str(out_dir / f'{prefix}_{safe}.png'), strip)

    # ── 요약 ───────────────────────────────────────────────────────────────────
    total_elapsed = total_timer.elapsed()
    log(f'\n{"="*70}')
    log(f'[결과 요약]')
    log(f'  정확도       : {correct_count}/{total_names} = {correct_count/total_names*100:.1f}%')
    log(f'  추론 총 시간 : {total_infer:.1f}s')
    log(f'  이름당 평균  : {total_infer/total_names:.2f}s')
    log(f'  전체 소요    : {total_elapsed:.1f}s  (모델 로딩 포함)')

    if failures:
        log(f'\n[오답 {len(failures)}개]')
        for gt_name, gt_jamos, pred_jamos, restored, _ in failures:
            log(f'  {gt_name} → {restored}')
            for i, (g, p) in enumerate(zip(gt_jamos, pred_jamos)):
                if g != p:
                    log(f'    자모[{i}]: GT={g}  Pred={p}')
        log(f'\n  오답 이미지: {out_dir}/FAIL_*.png')
    else:
        log('\n  오답 없음 — 전체 정답!')

    log(f'\n로그 저장: {log_path}')
    _LOG_FILE.close()


if __name__ == '__main__':
    main()
