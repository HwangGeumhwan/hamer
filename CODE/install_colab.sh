#!/bin/bash
# Colab 전용 HaMeR 환경 설치 스크립트
# - torch/torchvision: Colab 기설치 버전 그대로 사용
# - detectron2: 설치된 torch 버전에 맞는 prebuilt wheel 자동 선택, 없으면 소스 빌드
# - mmcv-full 1.5.0: 소스 빌드
set -e

HAMER_DIR="${HAMER_DIR:-/content/hamer}"

echo "==== Colab HaMeR 설치 시작 ===="
echo "  HAMER_DIR: $HAMER_DIR"
nvidia-smi | head -4 2>/dev/null || echo "  (nvidia-smi 없음)"

# 설치된 torch 버전 및 GPU 아키텍처 감지
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
TORCH_MAJOR=$(python3 -c "import torch; print(torch.__version__.split('.')[0])" 2>/dev/null || echo "2")
TORCH_MINOR=$(python3 -c "import torch; print(torch.__version__.split('.')[1])" 2>/dev/null || echo "0")
ARCH=$(python3 -c "
import torch
cap = torch.cuda.get_device_capability()
print(f'{cap[0]}.{cap[1]}')
" 2>/dev/null || echo "7.5")
echo "  torch: $TORCH_VER  CUDA arch: $ARCH"

# ── [1/7] setuptools ──────────────────────────────────────────────────────────
echo "==== [1/7] setuptools ===="
pip install -q "setuptools>=59.5.0" wheel

# ── [2/7] numpy 고정 ──────────────────────────────────────────────────────────
echo "==== [2/7] numpy 1.26.4 고정 ===="
# 충돌 경고는 jax/rasterio 등 미사용 패키지 — 무시해도 됨
pip install -q "numpy==1.26.4"

# ── [3/7] detectron2 ──────────────────────────────────────────────────────────
echo "==== [3/7] detectron2 ===="
# 설치된 torch major.minor 에 맞는 prebuilt wheel 순서대로 시도
INSTALLED=false
for TORCH_TAG in "torch${TORCH_MAJOR}.${TORCH_MINOR}" "torch${TORCH_MAJOR}.0" "torch2.2" "torch2.1" "torch2.0"; do
    for CU_TAG in "cu121" "cu118" "cu117"; do
        URL="https://dl.fbaipublicfiles.com/detectron2/wheels/${CU_TAG}/${TORCH_TAG}/index.html"
        if pip install -q detectron2 -f "$URL" 2>/dev/null; then
            echo "  -> prebuilt 성공: ${CU_TAG}/${TORCH_TAG}"
            INSTALLED=true
            break 2
        fi
    done
done

if [ "$INSTALLED" = false ]; then
    echo "  -> prebuilt 없음, 소스 빌드..."
    TORCH_INC=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
    CXXFLAGS="-I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include" \
    CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="$ARCH" \
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
fi

# ── [4/7] mmcv-full 1.5.0 소스 빌드 ─────────────────────────────────────────
echo "==== [4/7] mmcv-full 1.5.0 소스 빌드 ===="
TORCH_INC=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
CUDA_HOME_VAL=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null | head -1 || echo "/usr/local/cuda")
# CUDA_HOME은 실제 존재하는 경로로
for _cu in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-13 /usr/local/cuda-11.8; do
    [ -d "$_cu" ] && CUDA_HOME_VAL="$_cu" && break
done
echo "  CUDA_HOME: $CUDA_HOME_VAL"

SETUPTOOLS_USE_DISTUTILS=local \
CXXFLAGS="-I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include -I$TORCH_INC/TH -I$TORCH_INC/THC" \
CUDA_HOME="$CUDA_HOME_VAL" \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="$ARCH" \
MMCV_WITH_OPS=1 \
pip install --no-build-isolation 'mmcv-full==1.5.0' 2>&1 | tee /tmp/mmcv_build.log || {
    echo "=== mmcv-full 빌드 실패 — 에러 마지막 50줄 ==="
    tail -50 /tmp/mmcv_build.log
    exit 1
}

# ── [5/7] 버전 고정 패키지 ───────────────────────────────────────────────────
echo "==== [5/7] 버전 고정 패키지 ===="
# tensorflow: Python 3.12 미지원 + 추론 코드에서 미사용 → 제외
# smplx: hamer 의존성, pip install -e 실패 시 누락될 수 있어 명시적 설치
pip install -q \
    "timm==0.6.13" \
    "pytorch-lightning==2.0.3" \
    "scikit-learn" \
    "mediapipe" \
    "joblib" \
    "smplx"

# ── [6/7] chumpy + numpy 2.x 호환 패치 ──────────────────────────────────────
echo "==== [6/7] chumpy 설치 및 numpy 2.x 호환 패치 ===="
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

# ── [7/7] hamer + ViTPose 설치 ───────────────────────────────────────────────
echo "==== [7/7] hamer + ViTPose 설치 ===="
cd "$HAMER_DIR"
pip install -q -e ".[all]"
pip install -q -e third-party/ViTPose

echo "  -> numpy/opencv 최종 고정"
pip install -q "numpy==1.26.4" "opencv-python==4.8.1.78"

# ── 검증 ──────────────────────────────────────────────────────────────────────
echo "==== 검증 ===="
python3 -c "
import torch, detectron2, mmcv, timm
import pytorch_lightning as pl, sklearn, cv2, mediapipe as mp
print(f'python           : $(python3 --version)')
print(f'torch            : {torch.__version__}')
print(f'CUDA available   : {torch.cuda.is_available()}')
print(f'detectron2       : {detectron2.__version__}')
print(f'mmcv             : {mmcv.__version__}')
print(f'timm             : {timm.__version__}')
print(f'pytorch_lightning : {pl.__version__}')
print(f'opencv           : {cv2.__version__}')
print(f'mediapipe        : {mp.__version__}')
" 2>&1 | grep -v "dlerror\|cudart\|Ignore\|LD_LIBRARY"

echo ""
echo "설치 완료! Colab 런타임을 재시작하세요."
