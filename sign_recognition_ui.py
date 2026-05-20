"""
수화 글자 인식 - UI 버전
HaMeR 키포인트 검출 + SLT TFLite LSTM 분류기

Usage:
    python sign_recognition_ui.py --video path/to/video.mp4
    python sign_recognition_ui.py --video 0          # 웹캠
"""
import argparse
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel

# ── 상수 ──────────────────────────────────────────────────────────────────────
ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
]
SEQ_LENGTH = 10
CONF_THRESHOLD = 0.9
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf'

# 손 관절 연결 (MediaPipe/MANO 21-keypoint 기준)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


# ── 피처 추출 ─────────────────────────────────────────────────────────────────
def vector_normalization(joint):
    """(21,2) 손 관절 → 55-dim 피처 (벡터 40 + 각도 15)"""
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
    v = v2 - v1
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norms > 1e-8, norms, 1e-8)
    angle = np.arccos(np.clip(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]), -1.0, 1.0))
    return v, np.degrees(angle).astype(np.float32)


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_models(checkpoint, body_detector='regnety'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'[INFO] device: {device}')

    download_models(CACHE_DIR_HAMER)
    hamer_model, model_cfg = load_hamer(checkpoint)
    hamer_model = hamer_model.to(device).eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer as _hamer
        cfg_path = Path(_hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
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
        det_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(det_cfg)

    cpm = ViTPoseModel(device)
    return hamer_model, model_cfg, detector, cpm, device


# ── 프레임 처리 ───────────────────────────────────────────────────────────────
def process_frame(img_cv2, hamer_model, model_cfg, detector, cpm, device, rescale_factor=2.0):
    """
    Returns:
        feature  : np.ndarray (55,) or None  — TFLite 입력용
        kp2d_img : np.ndarray (21,2) or None — 화면 시각화용 ViTPose 2D 좌표
    """
    # 1. 사람 검출
    det_out = detector(img_cv2)
    instances = det_out['instances']
    valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
    pred_bboxes = instances.pred_boxes.tensor[valid].cpu().numpy()
    pred_scores = instances.scores[valid].cpu().numpy()
    if len(pred_bboxes) == 0:
        return None, None

    # 2. ViTPose 손 키포인트 검출 — 오른손만 사용
    img_rgb = img_cv2[:, :, ::-1].copy()
    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes, rights, kp2d_img = [], [], None
    for vitposes in vitposes_out:
        right_keyp = vitposes['keypoints'][-21:]          # 오른손 21개
        valid_kp = right_keyp[:, 2] > 0.5
        if valid_kp.sum() > 3:
            bbox = [
                right_keyp[valid_kp, 0].min(), right_keyp[valid_kp, 1].min(),
                right_keyp[valid_kp, 0].max(), right_keyp[valid_kp, 1].max(),
            ]
            bboxes.append(bbox)
            rights.append(1)
            kp2d_img = right_keyp[:, :2]                  # 시각화용

    if not bboxes:
        return None, None

    # 3. HaMeR 추론
    dataset = ViTDetDataset(
        model_cfg, img_cv2,
        np.stack(bboxes), np.stack(rights),
        rescale_factor=rescale_factor,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = hamer_model(batch)

        for n in range(batch['img'].shape[0]):
            if not batch['right'][n].cpu().numpy():       # 오른손만
                continue
            kp3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21,3)
            vector, angle = vector_normalization(kp3d[:, :2])
            feature = np.concatenate([vector.flatten(), angle.flatten()])  # 55-dim
            return feature, kp2d_img

    return None, None


# ── UI 렌더링 ─────────────────────────────────────────────────────────────────
def draw_overlay(img, seq_len, last_action, conf, kp2d_img, font):
    h, w = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 시퀀스 진행 바
    bar_w, bar_h = int(w * 0.35), 18
    bx, by = 10, 10
    fill_w = int(bar_w * seq_len / SEQ_LENGTH)
    draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=(40, 40, 40))
    draw.rectangle([bx, by, bx + fill_w, by + bar_h], fill=(50, 200, 100))
    draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(180, 180, 180))
    draw.text((bx + bar_w + 6, by), f'{seq_len}/{SEQ_LENGTH}', fill=(220, 220, 220))

    # 인식 결과
    if last_action:
        draw.text((10, 40), f'인식: {last_action}', font=font, fill=(50, 255, 120))
        draw.text((10, 90), f'conf: {conf:.2f}', fill=(200, 200, 200))

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ViTPose 2D 키포인트 오버레이
    if kp2d_img is not None:
        pts = kp2d_img.astype(int)
        for a, b in HAND_CONNECTIONS:
            if 0 <= pts[a, 0] < w and 0 <= pts[b, 0] < w:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (100, 200, 255), 1)
        for pt in pts:
            cv2.circle(img, tuple(pt), 4, (255, 220, 50), -1)

    return img


# ── 비동기 추론 상태 ──────────────────────────────────────────────────────────
class _InferState:
    def __init__(self):
        self.lock    = threading.Lock()
        self.busy    = False
        self.feature = None   # 새 피처 (소비 전)
        self.kp2d    = None   # 마지막으로 검출된 키포인트 (표시 유지용)

    def submit(self, frame, hamer_model, model_cfg, detector, cpm, device):
        """추론 스레드 시작. 이미 실행 중이면 건너뜀."""
        with self.lock:
            if self.busy:
                return
            self.busy = True
        threading.Thread(
            target=self._worker,
            args=(frame, hamer_model, model_cfg, detector, cpm, device),
            daemon=True,
        ).start()

    def _worker(self, frame, hamer_model, model_cfg, detector, cpm, device):
        feat, kp2d = process_frame(frame, hamer_model, model_cfg, detector, cpm, device)
        with self.lock:
            self.feature = feat
            if kp2d is not None:
                self.kp2d = kp2d   # 검출 성공 시에만 갱신
            self.busy = False

    def consume(self):
        """새 피처가 있으면 반환 후 초기화. 없으면 None."""
        with self.lock:
            feat = self.feature
            self.feature = None
        return feat

    def last_kp2d(self):
        with self.lock:
            return self.kp2d

    def is_busy(self):
        with self.lock:
            return self.busy


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run(args):
    hamer_model, model_cfg, detector, cpm, device = load_models(
        args.checkpoint, args.body_detector)

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    font = ImageFont.truetype(FONT_PATH, 48) if os.path.exists(FONT_PATH) else ImageFont.load_default()

    video_src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise RuntimeError(f'영상을 열 수 없습니다: {video_src}')

    # 웹캠은 열린 직후 프레임이 준비되지 않아 read()가 실패함 — 초기화 대기
    if isinstance(video_src, int):
        time.sleep(1.0)
        for _ in range(5):   # 버퍼 플러시
            cap.read()

    seq        = deque(maxlen=SEQ_LENGTH)
    action_seq = deque(maxlen=3)
    last_action, last_conf = None, 0.0
    infer = _InferState()

    print('[INFO] ESC 키로 종료')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 추론 스레드가 놀고 있으면 현재 프레임 제출
        infer.submit(frame.copy(), hamer_model, model_cfg, detector, cpm, device)

        # 새 피처가 도착했으면 시퀀스에 추가 → TFLite 예측
        feature = infer.consume()
        if feature is not None:
            seq.append(feature)
            if len(seq) == SEQ_LENGTH:
                x = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                interpreter.set_tensor(in_det[0]['index'], x)
                interpreter.invoke()
                y_pred = interpreter.get_tensor(out_det[0]['index'])[0]
                i_pred = int(np.argmax(y_pred))
                conf   = float(y_pred[i_pred])
                if conf >= CONF_THRESHOLD:
                    action = ACTIONS[i_pred]
                    action_seq.append(action)
                    if (len(action_seq) == 3
                            and action_seq[-1] == action_seq[-2] == action_seq[-3]):
                        last_action = action
                        last_conf   = conf

        # 매 프레임 즉시 표시 (추론 완료를 기다리지 않음)
        display = draw_overlay(frame, len(seq), last_action, last_conf,
                               infer.last_kp2d(), font)
        cv2.imshow('Sign Language Recognition (ESC: 종료)', display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='수화 글자 인식 UI')
    parser.add_argument('--video', type=str, default='0',
                        help='비디오 파일 경로 또는 웹캠 인덱스 (기본: 0)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--tflite', type=str,
                        default='Sign_Language_Translation/models/multi_hand_gesture_classifier.tflite')
    parser.add_argument('--body_detector', type=str, default='regnety',
                        choices=['vitdet', 'regnety'])
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
