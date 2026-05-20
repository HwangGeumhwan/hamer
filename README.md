# HaMeR: Hand Mesh Recovery
Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**

[Georgios Pavlakos](https://geopavlakos.github.io/), [Dandan Shan](https://ddshan.github.io/), [Ilija Radosavovic](https://people.eecs.berkeley.edu/~ilija/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [David Fouhey](https://cs.nyu.edu/~fouhey/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05251-00ff00.svg)](https://arxiv.org/pdf/2312.05251.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://geopavlakos.github.io/hamer/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rQbQzegFWGVOm1n1d-S6koOWDo7F2ucu?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/geopavlakos/HaMeR)

![teaser](assets/teaser.jpg)

## News

- [2026/02] Check out our [new work](https://gkarv.github.io/hand-texture-module/) using learned texture priors to improve HaMeR.
- [2024/06] HaMeR received the 2nd place award in the Ego-Pose Hands task of the Ego-Exo4D Challenge! Please check the [validation report](https://www.cs.utexas.edu/~pavlakos/hamer/resources/egoexo4d_challenge.pdf).
- [2024/05] We have released the evaluation pipeline!
- [2024/05] We have released the HInt dataset annotations! Please check [here](https://github.com/ddshan/hint).
- [2023/12] Original release!

## 설치

### 1. 시스템 요구사항

- Python 3.10
- Anaconda(알아서 설치. 설치시 Add to PATH 무조건 체크. 도커 쓸거면 필요없긴 함)
- NVIDIA 드라이버 520 이상


### 2. 저장소 클론

```bash
git clone --recursive https://github.com/HwangGeumhwan/hamer
cd hamer
```

### 3. 설치. 아래 셋 중 하나를 선택할 것
#### 3-1. 자동 설치 (권장)

아래 스크립트가 conda 환경 생성부터 모든 의존성 설치까지 자동으로 처리합니다.

```bash
bash install_hamer.sh
conda activate hamer
```

#### 수동 설치 (비권장)

자동 스크립트 대신 단계별로 직접 설치하려면:

```bash
conda create -n hamer python=3.10 -y
conda activate hamer

# setuptools 버전 고정 (C++ 확장 빌드 호환성)
pip install "setuptools==59.5.0" wheel

# PyTorch (CUDA 12.x 드라이버와 호환)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# detectron2 prebuilt wheel (PyPI 미등록, 직접 설치 필요)
pip install detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html

# mmcv-full 1.x (ViTPose 의존성, mmcv 2.x와 다른 패키지)
pip install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# 버전 고정 패키지
pip install "timm==0.6.13" "pytorch-lightning==1.9.5"

# chumpy (격리 빌드 우회 및 numpy 2.x 호환 패치 필요)
pip install --no-build-isolation chumpy
CHUMPY_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/chumpy
sed -i \
    -e 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' \
    -e 's/np\.bool\b/np.bool_/g' \
    -e 's/np\.int\b/np.int_/g' \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.complex\b/np.complex128/g' \
    "$CHUMPY_DIR/__init__.py" "$CHUMPY_DIR"/*.py

# hamer 및 ViTPose
pip install -e ".[all]"
pip install -v -e third-party/ViTPose
```

#### Docker Compose (안해봄)

If you wish to use HaMeR with Docker, you can use the following command:

```
docker compose -f ./docker/docker-compose.yml up -d
```

After the image is built successfully, enter the container and run the steps as above:

```
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

Continue with the installation steps:

```bash
bash fetch_demo_data.sh
```


### 모델 다운로드

학습된 모델과 데모 데이터를 다운로드합니다 (`_DATA/` 폴더에 저장됩니다):

```bash
bash fetch_demo_data.sh
```

MANO 모델은 별도로 다운로드해야 합니다. [MANO 웹사이트](https://mano.is.tue.mpg.de)에서 가입 -> Models & Code 다운로드

압축 해제 후, 오른손 모델(`ano_v1_2/models/MANO_RIGHT.pkl`)을 받아 아래 경로에 배치:

```
_DATA/data/mano/MANO_RIGHT.pkl
```


## Demo (실행해보기)
```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

---

## 한국어 지문자 인식 파이프라인

HaMeR를 3D 피처 추출기로 활용하여 한국어 지문자(자음 14자 + 모음 17자, 총 31클래스)를 인식하는 파이프라인입니다.  
MediaPipe 2D 랜드마크 대신 HaMeR의 3D 관절 벡터/각도를 피처로 사용합니다.

### 파이프라인 개요

```
영상 (.avi/.mp4)
    ↓ extract_hamer_features.py
PNG 스켈레톤 + NPZ 피처 (_DATA/data/hamer_features/)
    ↓ dataset_hamer.py
train_data.npz  (x: N×35,  y: N)
    ↓ train_hamer.py
gesture_mlp.pth  (학습된 MLP 분류기)
    ↓ eval_hamer.py
result/  (혼동 행렬 히트맵 | 라벨별 정확도 | 오답 이미지)

[선택] 이미지 시퀀스 / 텍스트 → 한국어 이름 복원
    ↓ eval_hamer.py --restore_images / --restore_text
    ↓ korean_name_postprocess.py
result/  (복원 결과 이미지 + 콘솔 출력)
```

### 데이터 디렉토리 구조

```
_DATA/data/hamer_features/
├── png/
│   └── {label}/            # 프레임 PNG (입력 패치 | 스켈레톤)
│       └── {stem}_f{idx:04d}.png
├── mp4/
│   └── {label}/            # PNG를 합친 스켈레톤 영상
│       └── {stem}.mp4
├── npz/
│   └── {label}/            # 프레임별 + 시퀀스 NPZ
│       ├── {stem}_f{idx:04d}.npz   # keypoints_3d(21,3), vectors(20,3), angles(15,), features(75,)
│       └── {stem}_seq.npz          # features(N,75), global_orients(N,9), frame_indices, label_idx, label_str
├── train_data.npz           # 학습용 데이터 (x: N×35, y: N)
├── eval_data.npz            # 평가용 데이터 (x: N×35, y: N)
└── model/
    ├── gesture_mlp.pth      # 학습된 모델
    └── train_curve.png      # 학습 곡선
```

### 피처 설명

#### 75-dim 전체 피처 (`features`)
- **벡터 60-dim**: 손가락 관절 간 방향벡터 20개 × 3D = 60
- **각도 15-dim**: 인접 뼈대 사이 굽힘 각도 15개 (단위: degrees)

#### 35-dim MLP 입력 피처
| 인덱스 | 구성 | 설명 |
|--------|------|------|
| 0–2   | angles[0:3]    | 엄지 전체 굽힘 각도 (MCP/PIP/DIP) |
| 3–4   | angles[3:5]    | 검지 MCP·PIP 각도 |
| 5–6   | angles[6:8]    | 중지 MCP·PIP 각도 |
| 7–8   | angles[9:11]   | 약지 MCP·PIP 각도 |
| 9–10  | angles[12:14]  | 소지 MCP·PIP 각도 |
| 11–13 | vectors[3]     | 엄지 DIP→TIP 방향벡터 (3D) |
| 14–16 | vectors[7]     | 검지 DIP→TIP 방향벡터 (3D) |
| 17–19 | vectors[11]    | 중지 DIP→TIP 방향벡터 (3D) |
| 20–22 | vectors[15]    | 약지 DIP→TIP 방향벡터 (3D) |
| 23–25 | vectors[19]    | 소지 DIP→TIP 방향벡터 (3D) |
| 26–34 | global_orient  | MANO 손 전역 회전행렬 (3×3 flatten, 9D) |

### 실행 방법

#### Step 1. 피처 추출

입력 영상(`dataset/data_all/{label}/*.avi|*.mp4`)에서 HaMeR 3D 피처를 추출합니다.

```bash
python extract_hamer_features.py --video_root dataset/data_all --out_root _DATA/data/hamer_features --body_detector regnety
```

#### Step 2. 학습 데이터셋 생성

추출된 NPZ 파일의 각 프레임을 독립 샘플로 수집합니다. `--evalset`으로 지정한 번호의 데이터를 eval set으로 분리합니다(기본: 2번).

```bash
python dataset_hamer.py --npz_root _DATA/data/hamer_features/npz --out _DATA/data/hamer_features/train_data.npz --evalset 3 --eval_out _DATA/data/hamer_features/eval_data.npz
```

출력: `train_data.npz` — `x: (N, 35)`, `y: (N,)` / `eval_data.npz` — 동일 형식

#### Step 3. MLP 분류기 학습

```bash
python train_hamer.py --out_dir _DATA/data/hamer_features/model --batch_size 32 --lr 1e-3
```

`train_data.npz`로 훈련하고 `eval_data.npz`로 검증합니다 (Step 2에서 생성한 분할을 그대로 사용).  
모델 아키텍처: `Linear(35→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→31)`  
마지막 epoch 모델이 `gesture_mlp.pth`로 저장됩니다.

#### Step 4. 평가

```bash
python eval_hamer.py --npz_root _DATA/data/hamer_features/npz --png_root _DATA/data/hamer_features/png --evalset 3 --model _DATA/data/hamer_features/model/gesture_mlp.pth --out_dir result
```

출력: `result/confusion_matrix.png`, `result/accuracy_per_label.png`, `result/failed/`

결과 이미지: `{img_stem}_result.png` — 원본 | 스켈레톤 | 예측 글자(신뢰도, 정답) 패널

---

### 한국어 이름 복원

수화 이미지 시퀀스(자모 단위)를 입력받아 완성된 한국어 이름을 복원하는 후처리 파이프라인입니다.
후처리 로직은 `korean_name_postprocess.py`에 구현되어 있으며, `eval_hamer.py`에서 두 가지 모드로 실행합니다.

#### 복원 규칙

| 규칙 | 설명 |
|------|------|
| 완성 글자 | 각 음절은 초성 + 중성 [+ 종성]으로 구성된 완성형 |
| 겹받침 제외 | 종성은 단자음만 허용 |
| 쌍자음 | 동일 단자음 두 번 연속 입력 → 쌍자음 (ㄱ+ㄱ=ㄲ, ㄷ+ㄷ=ㄸ, ㅂ+ㅂ=ㅃ, ㅅ+ㅅ=ㅆ, ㅈ+ㅈ=ㅉ) |
| 이중모음 | 단모음 조합으로 입력 가능 (ㅗ+ㅏ=ㅘ, ㅗ+ㅐ=ㅙ, ㅜ+ㅓ=ㅝ, ㅜ+ㅔ=ㅞ 등) |

#### 모드 A — 이미지 시퀀스 입력 → 이름 복원

PNG 파일 목록을 순서대로 입력하면 각 이미지에서 자모를 예측하고 이름을 복원합니다.  
이미지 경로는 `png_root/{label}/{stem}_f{idx:04d}.png` 형식이어야 합니다.

```bash
python eval_hamer.py \
    --npz_root _DATA/data/hamer_features/npz \
    --model    _DATA/data/hamer_features/model/gesture_mlp.pth \
    --out_dir  result \
    --restore_images \
        _DATA/data/hamer_features/png/ㅎ/name1_f0003.png \
        _DATA/data/hamer_features/png/ㅗ/name1_f0007.png \
        _DATA/data/hamer_features/png/ㅇ/name1_f0012.png
```

출력: `result/restored_from_images.png` — 자모 패널 + 복원 결과

#### 모드 B — 글자 입력 → 데이터셋 무작위 추출 → 이름 복원

한국어 이름을 텍스트로 입력하면 자모로 분해 후 각 자모에 대응하는 데이터셋 샘플을 무작위로 선택하여 예측하고 복원합니다.

```bash
python eval_hamer.py \
    --npz_root _DATA/data/hamer_features/npz \
    --png_root _DATA/data/hamer_features/png \
    --model    _DATA/data/hamer_features/model/gesture_mlp.pth \
    --out_dir  result \
    --restore_text "홍길동" \
    --seed 42
```

출력: `result/restored_홍길동.png` — 무작위 선택된 자모 이미지 패널 + 복원 결과 (원본 대조)

#### API 직접 사용

```python
from korean_name_postprocess import parse_jamo_to_korean, decompose_korean_to_jamo

# 자모 시퀀스 → 이름
parse_jamo_to_korean(['ㄱ', 'ㅣ', 'ㅁ', 'ㅅ', 'ㅓ', 'ㅇ'])  # → '김성'
parse_jamo_to_korean(['ㄷ', 'ㄷ', 'ㅜ'])                    # → '뚜'  (쌍자음)
parse_jamo_to_korean(['ㅂ', 'ㅗ', 'ㅏ'])                    # → '봐'  (이중모음)

# 이름 → 자모 시퀀스 (역방향)
decompose_korean_to_jamo('김성')   # → ['ㄱ', 'ㅣ', 'ㅁ', 'ㅅ', 'ㅓ', 'ㅇ']
decompose_korean_to_jamo('홍길동') # → ['ㅎ', 'ㅗ', 'ㅇ', 'ㄱ', 'ㅣ', 'ㄹ', 'ㄷ', 'ㅗ', 'ㅇ']
```

---

## HInt Dataset
We have released the annotations for the HInt dataset. Please follow the instructions [here](https://github.com/ddshan/hint)

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Evaluation
Download the [evaluation metadata](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh) to `./hamer_evaluation_data/`. Additionally, download the FreiHAND, HO-3D, and HInt dataset images and update the corresponding paths in  `hamer/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`. 
```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC'
```

Results for HInt are stored in `results/eval_regression.csv`. For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) you get as output a `.json` file that can be used for evaluation using their corresponding evaluation processes.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Open-Source Contributions
- [Wentao Hu](https://vincenthu19.github.io/) integrated the hand parameters predicted by HaMeR into SMPL-X - [Mano2Smpl-X](https://github.com/VincentHu19/Mano2Smpl-X)

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
