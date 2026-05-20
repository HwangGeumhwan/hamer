"""
Video → HaMeR 3D feature extractor (메모리 절약 버전)

Input : dataset/data_all/{label}/*.avi|*.mp4
Output: _DATA/data/hamer_features/{label}/
          {stem}_f{idx:04d}.png  : 검출 성공 프레임 원본 (풀 프레임)
          {stem}_f{idx:04d}.npz  : 프레임별 3D 피처
            - keypoints_3d  : (21, 3)
            - vectors       : (20, 3)
            - angles        : (15,)
            - features      : (75,)   vector 60 + angle 15
            - global_orient : (9,)    MANO 손 전역 회전행렬 (3×3 flatten)
          {stem}_seq.npz         : 영상 전체 집계
            - features      : (N, 75)
            - global_orients: (N, 9)
            - frame_indices : (N,)
            - label_idx     : scalar
            - label_str     : str
"""

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
from datetime import datetime

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.render_openpose import render_openpose
from vitpose_model import ViTPoseModel

VIDEO_EXTS = {'.avi', '.mp4'}
ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
]

# 라벨별 앞쪽 스킵 프레임 수 (휴리스틱 보정)
LABEL_SKIP_FRAMES = {
    'ㅠ': 8,
}


def _vector_normalization_3d(joint):
    """(21, 3) → vectors (20,3), angles (15,)"""
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    angle = np.arccos(np.clip(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]]), -1.0, 1.0))
    return v, np.degrees(angle).astype(np.float32)



def detect_hands(img_cv2, detector, cpm):
    """Detectron2 + ViTPose로 손 바운딩박스 검출.
    Returns (bboxes, is_right, torso_center_2d, right_wrist_2d,
             right_shoulder_2d) or (None, None, None, None, None)."""
    det_out = detector(img_cv2)
    instances = det_out['instances']
    valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
    pred_bboxes = instances.pred_boxes.tensor[valid].cpu().numpy()
    pred_scores = instances.scores[valid].cpu().numpy()

    if len(pred_bboxes) == 0:
        return None, None, None, None, None

    vitposes_out = cpm.predict_pose(
        img_cv2[:, :, ::-1],
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes, sides     = [], []
    torso_center      = None
    right_wrist_2d    = None
    right_shoulder_2d = None

    for vp in vitposes_out:
        kps      = vp['keypoints']
        body_kps = kps[:-42]

        # 몸통 중심: 어깨(5,6) + 엉덩이(11,12) 평균
        if torso_center is None and len(body_kps) >= 13:
            pts = [body_kps[i][:2] for i in [5, 6, 11, 12]
                   if body_kps[i][2] > 0.3]
            if pts:
                torso_center = np.mean(pts, axis=0)

        # 오른쪽 어깨(6) 위치
        if right_shoulder_2d is None and len(body_kps) >= 7:
            r_sh = body_kps[6]
            if r_sh[2] > 0.3:
                right_shoulder_2d = r_sh[:2].copy()

        for keyp, side in [(kps[-42:-21], 0), (kps[-21:], 1)]:
            valid_kp = keyp[:, 2] > 0.5
            if valid_kp.sum() > 3:
                bbox = [keyp[valid_kp, 0].min(), keyp[valid_kp, 1].min(),
                        keyp[valid_kp, 0].max(), keyp[valid_kp, 1].max()]
                bboxes.append(bbox)
                sides.append(side)
                if side == 1 and right_wrist_2d is None and keyp[0][2] > 0.3:
                    right_wrist_2d = keyp[0][:2]

    if not bboxes:
        return None, None, None, None, None
    return np.stack(bboxes), np.array(sides), torso_center, right_wrist_2d, right_shoulder_2d


def _pngs_to_mp4(png_dir, mp4_path, stem, fps=10):
    frames = sorted(png_dir.glob(f'{stem}_f*.png'))
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()


def process_video(video_path, label_str, label_idx,
                  model, model_cfg, detector, cpm,
                  device, png_dir, npz_dir, mp4_dir, rescale_factor):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}")
        return

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 10
    stem       = video_path.stem
    skip_front = LABEL_SKIP_FRAMES.get(label_str, 0)

    seq_features       = []
    seq_global_orients = []
    seq_frame_idx      = []

    frame_idx  = 0
    detected   = 0
    blur_count = 0

    while True:
        ret, img_cv2 = cap.read()
        if not ret:
            break

        bboxes, sides, torso_center, right_wrist_2d, right_shoulder_2d = detect_hands(img_cv2, detector, cpm)

        if bboxes is not None:
            right_mask = sides == 1
            if right_mask.any():
                r_bboxes = bboxes[right_mask][:1]
                r_sides  = sides[right_mask][:1]

                dataset    = ViTDetDataset(model_cfg, img_cv2, r_bboxes, r_sides,
                                           rescale_factor=rescale_factor)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0)

                batch = next(iter(dataloader))
                if dataset.blur_applied:
                    blur_count += 1
                batch = recursive_to(batch, device)

                with torch.no_grad():
                    out = model(batch)

                kp3d       = out['pred_keypoints_3d'][0].detach().cpu().numpy()
                cam_t_crop = out['pred_cam_t'][0].detach().cpu().numpy()
                vector, angle = _vector_normalization_3d(kp3d)
                feature = np.concatenate([vector.flatten(), angle.flatten()])

                # MANO 전역 회전행렬 (3×3 → 9-dim flatten)
                global_orient = (out['pred_mano_params']['global_orient'][0]
                                 .detach().cpu().numpy()
                                 .reshape(9).astype(np.float32))

                # PNG: 항상 저장 (프레임 스킵 무관)
                input_patch = (batch['img'][0].cpu()
                               * (DEFAULT_STD[:, None, None] / 255)
                               + (DEFAULT_MEAN[:, None, None] / 255))
                input_patch = input_patch.permute(1, 2, 0).numpy()
                H_crop, W_crop = input_patch.shape[:2]
                f_crop = float(model_cfg.EXTRA.FOCAL_LENGTH)
                kp3d_cam = kp3d + cam_t_crop
                x_px = f_crop * kp3d_cam[:, 0] / kp3d_cam[:, 2] + W_crop / 2
                y_px = f_crop * kp3d_cam[:, 1] / kp3d_cam[:, 2] + H_crop / 2
                kp_with_conf = np.concatenate(
                    [np.stack([x_px, y_px], axis=1), np.ones((21, 1))], axis=1)
                skel_img = render_openpose((input_patch * 255).astype(np.uint8), kp_with_conf) / 255.
                final_img = np.concatenate([input_patch, skel_img], axis=1)
                cv2.imwrite(str(png_dir / f'{stem}_f{frame_idx:04d}.png'),
                            (final_img[:, :, ::-1] * 255).astype(np.uint8))

                # NPZ·feature 누적: skip_front 이후 프레임만
                kp2d = np.stack([x_px, y_px], axis=1).astype(np.float32)  # (21, 2) crop 좌표계
                crop_hw = np.array([H_crop, W_crop], dtype=np.int32)
                if frame_idx >= skip_front:
                    np.savez(
                        npz_dir / f'{stem}_f{frame_idx:04d}.npz',
                        keypoints_3d=kp3d,
                        vectors=vector,
                        angles=angle,
                        features=feature,
                        global_orient=global_orient,
                        kp2d=kp2d,
                        crop_hw=crop_hw,
                    )
                    seq_features.append(feature)
                    seq_global_orients.append(global_orient)
                    seq_frame_idx.append(frame_idx)

                detected += 1

                del out, batch, kp3d, cam_t_crop, vector, angle, feature, global_orient
                del input_patch, kp3d_cam, kp_with_conf, skel_img, final_img
                torch.cuda.empty_cache()

        frame_idx += 1

    cap.release()

    # MP4: 저장된 PNG 전체로 영상 생성 (스킵 없이)
    _pngs_to_mp4(png_dir, mp4_dir / f'{stem}.mp4', stem, fps=fps)

    if seq_features:
        np.savez(
            npz_dir / f'{stem}_seq.npz',
            features=np.stack(seq_features),
            global_orients=np.stack(seq_global_orients),
            frame_indices=np.array(seq_frame_idx),
            label_idx=np.array([label_idx]),
            label_str=label_str,
            total_frames=np.array([total]),
        )
        blur_ratio = blur_count / detected if detected else 0.0
        print(f"  {video_path.name}: {detected}/{total} 프레임 검출 "
              f"(feature {len(seq_features)}프레임, skip={skip_front}) → {stem}_seq.npz  "
              f"[blur {blur_count}/{detected} = {blur_ratio:.1%}]")
    else:
        print(f"  {video_path.name}: 오른손 미검출, 건너뜀")


def main():
    parser = argparse.ArgumentParser(description='HaMeR 3D feature extractor')
    parser.add_argument('--video_root', type=str,
                        default='dataset/data_all')
    parser.add_argument('--out_root', type=str,
                        default='_DATA/data/hamer_features')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--body_detector', type=str, default='regnety',
                        choices=['vitdet', 'regnety'],
                        help='regnety: 빠르고 메모리 적음 / vitdet: 정확하지만 무거움')
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='처리할 라벨 목록 (예: --labels ㄱ ㄴ ㄷ). 미지정 시 전체 처리')
    args = parser.parse_args()

    video_root = Path(args.video_root)
    out_root   = Path(args.out_root)

    # HaMeR 로드 — init_renderer=False 로 내부 SkeletonRenderer/MeshRenderer 생성 건너뜀
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint, init_renderer=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    print(f"Device: {device}")

    # Detectron2
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer as hamer_pkg
        cfg_path = Path(hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        det_cfg = LazyConfig.load(str(cfg_path))
        det_cfg.train.init_checkpoint = (
            'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
            'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
        )
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

    cpm = ViTPoseModel(device)

    label_dirs = sorted([d for d in video_root.iterdir() if d.is_dir()])
    if args.labels:
        label_dirs = [d for d in label_dirs if d.name in args.labels]
        print(f"\n필터: {args.labels} → {len(label_dirs)}개 라벨 폴더 처리\n")
    else:
        print(f"\n총 {len(label_dirs)}개 라벨 폴더 발견\n")

    for label_dir in label_dirs:
        label_str = label_dir.name
        label_idx = ACTIONS.index(label_str) if label_str in ACTIONS else -1

        video_files = sorted([f for f in label_dir.iterdir()
                               if f.suffix.lower() in VIDEO_EXTS])
        if not video_files:
            print(f"[{label_str}] 영상 없음, 건너뜀")
            continue

        png_dir = out_root / 'png' / label_str
        npz_dir = out_root / 'npz' / label_str
        mp4_dir = out_root / 'mp4' / label_str
        png_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)
        mp4_dir.mkdir(parents=True, exist_ok=True)

        print(f"{datetime.now().strftime('%H:%M:%S')} [{label_str}] (idx={label_idx}) {len(video_files)}개 영상")
        for video_path in video_files:
            process_video(
                video_path, label_str, label_idx,
                model, model_cfg, detector, cpm,
                device, png_dir, npz_dir, mp4_dir,
                rescale_factor=args.rescale_factor,
            )

    print("\n완료.")


if __name__ == '__main__':
    main()
