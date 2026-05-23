#!/bin/bash
# HaMeR 환경 설치 스크립트
# 시스템: CUDA 12.2, RTX 2080 Ti (compute 7.5)
# 전략: prebuilt wheel 우선, 소스 빌드 필요 시 conda cuda-toolkit 활용
# 별도 설치: detectron2 (PyPI 없음), chumpy (격리빌드 불가 + numpy 2.x 비호환)
#            mmcv-full 1.x (setup.py의 'mmcv'는 2.x를 끌어오므로 먼저 설치)
set -e

ENV_NAME="hamer"
CONDA_BASE=$(conda info --base)
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"
SITE_PACKAGES="$ENV_PATH/lib/python3.10/site-packages"

echo "==== [1/9] conda 환경 생성 (Python 3.10) ===="
conda create -n $ENV_NAME python=3.10 -y

echo "==== [2/9] setuptools 다운그레이드 (C++ 확장 호환성) ===="
conda run -n $ENV_NAME pip install "setuptools==59.5.0" wheel

echo "==== [3/9] PyTorch cu118 설치 (CUDA 12.2 드라이버 하위 호환) ===="
# numpy도 함께 고정: torchvision이 numpy 2.x를 끌어오는 것을 방지
conda run -n $ENV_NAME pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    "numpy==1.26.4" \
    --index-url https://download.pytorch.org/whl/cu118

echo "==== [4/9] detectron2 설치 ===="
echo "  -> cu118/torch2.0 prebuilt wheel 시도..."
if conda run -n $ENV_NAME pip install detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html; then
    echo "  -> detectron2 cu118 prebuilt 설치 성공"
else
    echo "  -> cu118 실패, cu117 prebuilt wheel 시도..."
    if conda run -n $ENV_NAME pip install detectron2 \
        -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html; then
        echo "  -> detectron2 cu117 prebuilt 설치 성공"
    else
        echo "  -> prebuilt 실패, conda CUDA 11.8 toolkit 설치 후 소스 빌드..."
        conda install -n $ENV_NAME -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
        conda run -n $ENV_NAME bash -c "
            TORCH_INC=\$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"include\"))')
            export CXXFLAGS=\"-I\$TORCH_INC -I\$TORCH_INC/torch/csrc/api/include\"
            export CUDA_HOME=$ENV_PATH
            export FORCE_CUDA=1
            export TORCH_CUDA_ARCH_LIST='7.5'
            pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
        "
    fi
fi

echo "==== [5/9] numpy 1.x 고정 및 mmcv-full 1.5.0 소스 빌드 ===="
# numpy 2.x는 xtcocotools 등 1.x로 컴파일된 패키지와 binary 비호환
# mmcv-full 1.5.0: ViTPose가 요구하는 버전 (<=1.5.0), torch 2.0용 prebuilt wheel 없어 소스 빌드
# gcc로 C++ 파일 빌드 시 torch include path가 자동 추가 안 됨 → CXXFLAGS로 명시
conda run -n $ENV_NAME pip install "numpy==1.26.4"
conda install -n $ENV_NAME -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda run -n $ENV_NAME bash -c "
    TORCH_INC=\$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"include\"))')
    export CXXFLAGS=\"-I\$TORCH_INC -I\$TORCH_INC/torch/csrc/api/include -I\$TORCH_INC/TH -I\$TORCH_INC/THC\"
    CUDA_HOME=$ENV_PATH \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST='7.5' \
    MMCV_WITH_OPS=1 \
    pip install --no-build-isolation 'mmcv-full==1.5.0'
"

echo "==== [6/9] 버전 고정 패키지 사전 설치 ===="
# timm: hamer는 timm.models.layers API 사용 (0.6.x 기준 코드)
#       0.9+ 에서도 호환 alias 존재하나 0.6.13으로 고정하여 안전성 확보
# pytorch-lightning: hamer는 pytorch_lightning 임포트 사용 (v2.0에서 경로 변경됨)
# tensorflow 2.8.0: Python 3.10 지원 최소 버전
# protobuf 3.20.3: tensorflow 2.8.0이 protobuf 4+ 와 비호환 (Descriptors cannot be created directly)
conda run -n $ENV_NAME pip install \
    "timm==0.6.13" \
    "pytorch-lightning==2.0.3" \
    "tensorflow==2.8.0" \
    "protobuf==3.20.3" \
    "scikit-learn" \
    "mediapipe"

echo "==== [7/9] chumpy 설치 및 numpy 2.x 호환 패치 ===="
# --ignore-installed: ~/.local 캐시 무시하고 conda env에 설치
# --no-deps: scipy가 numpy 2.x를 끌어오는 것을 방지 (scipy는 ~/.local에 이미 존재)
conda run -n $ENV_NAME pip install --no-build-isolation --ignore-installed --no-deps chumpy

echo "  -> chumpy numpy 2.x 호환 패치 적용..."
# import로 탐지하면 numpy 2.x 환경에서 chumpy 자체가 깨져 exit code 1 → set -e 로 스크립트 종료
# SITE_PACKAGES는 스크립트 상단에 이미 정의됨 ($ENV_PATH/lib/python3.10/site-packages)
CHUMPY_DIR="$SITE_PACKAGES/chumpy"
if [ -d "$CHUMPY_DIR" ]; then
    for f in "$CHUMPY_DIR"/*.py; do
        sed -i \
            -e 's/np\.bool\b/np.bool_/g' \
            -e 's/np\.int\b/np.int_/g' \
            -e 's/np\.float\b/np.float64/g' \
            -e 's/np\.complex\b/np.complex128/g' \
            "$f"
    done
    sed -i \
        's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' \
        "$CHUMPY_DIR/__init__.py"
    echo "  -> 패치 완료: $CHUMPY_DIR"
else
    echo "  -> 경고: chumpy 디렉터리를 찾지 못했습니다"
fi

echo "==== [8/9] hamer 패키지 및 ViTPose 설치 ===="
conda run -n $ENV_NAME pip install -e ".[all]"
conda run -n $ENV_NAME pip install -v -e third-party/ViTPose

echo "  -> numpy/opencv 최종 버전 고정 (ViTPose/mmpose가 numpy 2.x를 끌어올 수 있음)"
conda run -n $ENV_NAME pip install "numpy==1.26.4" "opencv-python==4.8.1.78"

echo "==== [9/9] 설치 검증 ===="
conda run -n $ENV_NAME python -c "
import torch
import detectron2
import mmcv
import chumpy
import timm
import pytorch_lightning as pl
import tensorflow as tf
import sklearn
import cv2
import mediapipe as mp
print(f'torch           : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
print(f'detectron2      : {detectron2.__version__}')
print(f'mmcv            : {mmcv.__version__}')
print(f'chumpy          : {chumpy.__version__}')
print(f'timm            : {timm.__version__}')
print(f'pytorch_lightning: {pl.__version__}')
print(f'tensorflow      : {tf.__version__}')
print(f'scikit-learn    : {sklearn.__version__}')
print(f'opencv          : {cv2.__version__}')
print(f'mediapipe       : {mp.__version__}')
" 2>&1 | grep -v "dlerror\|cudart\|Ignore\|LD_LIBRARY"

echo ""
echo "설치 완료! 환경 활성화: conda activate $ENV_NAME"
