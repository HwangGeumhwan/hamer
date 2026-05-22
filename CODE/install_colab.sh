#!/bin/bash
# Colab 전용 HaMeR 환경 설치 스크립트
# install_hamer.sh 와 패키지 버전 완전 동일 (conda → pip 직접 설치로만 차이)
# CUDA: Colab GPU 아키텍처 자동 감지
set -e

HAMER_DIR="${HAMER_DIR:-/content/hamer}"

echo "==== Colab HaMeR 설치 시작 ===="
echo "  HAMER_DIR: $HAMER_DIR"
nvidia-smi | head -4

# GPU 아키텍처 자동 감지
ARCH=$(python3 -c "
import torch
cap = torch.cuda.get_device_capability()
print(f'{cap[0]}.{cap[1]}')
" 2>/dev/null || echo "7.5")
echo "  CUDA arch: $ARCH"

echo "==== [1/8] setuptools 다운그레이드 ===="
pip install -q "setuptools==59.5.0" wheel

echo "==== [2/8] PyTorch cu118 (로컬과 동일 버전) ===="
pip install -q \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    "numpy==1.26.4" \
    --index-url https://download.pytorch.org/whl/cu118

echo "==== [3/8] detectron2 ===="
if pip install -q detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html; then
    echo "  -> cu118 prebuilt 성공"
elif pip install -q detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html; then
    echo "  -> cu117 prebuilt 성공"
else
    echo "  -> 소스 빌드..."
    TORCH_INC=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
    CXXFLAGS="-I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include" \
    CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="$ARCH" \
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
fi

echo "==== [4/8] mmcv-full 1.5.0 소스 빌드 ===="
pip install -q "numpy==1.26.4"
TORCH_INC=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
CXXFLAGS="-I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include -I$TORCH_INC/TH -I$TORCH_INC/THC" \
CUDA_HOME="/usr/local/cuda" \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="$ARCH" \
MMCV_WITH_OPS=1 \
pip install --no-build-isolation 'mmcv-full==1.5.0'

echo "==== [5/8] 버전 고정 패키지 ===="
pip install -q \
    "timm==0.6.13" \
    "pytorch-lightning==2.0.3" \
    "tensorflow==2.8.0" \
    "protobuf==3.20.3" \
    "scikit-learn" \
    "mediapipe" \
    "joblib"

echo "==== [6/8] chumpy + numpy 2.x 호환 패치 ===="
pip install -q --no-build-isolation --ignore-installed --no-deps chumpy

CHUMPY_DIR=$(python3 -c "import chumpy, os; print(os.path.dirname(chumpy.__file__))" 2>/dev/null)
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
    echo "  -> chumpy 패치 완료"
fi

echo "==== [7/8] hamer + ViTPose 설치 ===="
cd "$HAMER_DIR"
pip install -q -e ".[all]"
pip install -q -v -e third-party/ViTPose

echo "  -> numpy/opencv 최종 고정"
pip install -q "numpy==1.26.4" "opencv-python==4.8.1.78"

echo "==== [8/8] 검증 ===="
python3 -c "
import torch, detectron2, mmcv, chumpy, timm
import pytorch_lightning as pl, sklearn, cv2, mediapipe as mp
print(f'torch            : {torch.__version__}')
print(f'CUDA available   : {torch.cuda.is_available()}')
print(f'detectron2       : {detectron2.__version__}')
print(f'mmcv             : {mmcv.__version__}')
print(f'timm             : {timm.__version__}')
print(f'pytorch_lightning : {pl.__version__}')
print(f'scikit-learn     : {sklearn.__version__}')
print(f'opencv           : {cv2.__version__}')
print(f'mediapipe        : {mp.__version__}')
" 2>&1 | grep -v "dlerror\|cudart\|Ignore\|LD_LIBRARY"

echo ""
echo "설치 완료! Colab 런타임을 재시작하세요."
