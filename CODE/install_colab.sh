#!/bin/bash
# Colab 전용 HaMeR 환경 설치 스크립트
# 로컬(install_hamer.sh) 환경 기준 — Colab에서 불가피한 부분만 변경:
#   - conda 없음 → pip 직접 사용
#   - torch/torchvision: Colab 기설치 버전 그대로 사용 (로컬: 2.0.1+cu118)
#   - CUDA_HOME: /usr/local/cuda (로컬: conda env)
#   - tensorflow: Python 3.11 미지원 버전 → 제외 (추론 코드에서 미사용)
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

# CUDA_HOME 설정 (nvcc 없어도 디렉터리 기준으로 설정)
CUDA_HOME=""
for _cu in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12 \
           /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    [ -d "$_cu" ] && CUDA_HOME="$_cu" && break
done
[ -z "$CUDA_HOME" ] && CUDA_HOME="/usr/local/cuda"
export CUDA_HOME CUDA_PATH="$CUDA_HOME"
# nvcc 탐색 후 PATH 추가
NVCC_PATH=$(find /usr/local/cuda* /usr/bin /usr/local/bin -name nvcc -type f 2>/dev/null | head -1)
[ -n "$NVCC_PATH" ] && export PATH="$(dirname $NVCC_PATH):$PATH"
echo "  CUDA_HOME: $CUDA_HOME  nvcc: $(which nvcc 2>/dev/null || echo 'not in PATH')"

# ── torch/utils/cpp_extension.py 직접 패치 ─────────────────────────────────
# pip subprocess는 환경변수를 상속하지 않으므로 소스 파일에 CUDA_HOME fallback 삽입
# (패치 후 .pyc 캐시는 mtime 갱신으로 자동 재컴파일됨)
python3 << 'PYEOF'
import os, re, inspect
import torch.utils.cpp_extension as e

src = inspect.getfile(e)
if '__pycache__' in src:
    src = re.sub(r'/__pycache__/(\w+)\.cpython-\d+\.pyc$', r'/\1.py', src)
src = re.sub(r'\.pyc$', '.py', src)

with open(src) as f:
    code = f.read()

TARGET = 'CUDA_HOME = _find_cuda_home()'
PATCH = """
# colab-patch: nvcc 없어도 디렉터리 존재하면 CUDA_HOME 설정
if CUDA_HOME is None:
    for _p in ['/usr/local/cuda', '/usr/local/cuda-13', '/usr/local/cuda-12']:
        if os.path.isdir(_p):
            CUDA_HOME = _p
            os.environ.setdefault('CUDA_HOME', _p)
            os.environ.setdefault('CUDA_PATH', _p)
            break"""

if TARGET in code and 'colab-patch' not in code:
    with open(src, 'w') as f:
        f.write(code.replace(TARGET, TARGET + PATCH, 1))
    print(f'  torch cpp_extension 패치 완료: {src}')
else:
    print(f'  torch cpp_extension 이미 패치됨')
PYEOF

# ── [1/7] setuptools ──────────────────────────────────────────────────────────
echo "==== [1/7] setuptools ===="
# 로컬은 59.5.0이나, Colab Python 3.11+에서는 detectron2 소스 빌드가 >=60 요구
# mmcv-full은 SETUPTOOLS_USE_DISTUTILS=local로 버전 무관하게 빌드 가능
pip install -q "setuptools>=60.0.0,<70.0.0" wheel

# ── [2/7] numpy 고정 ──────────────────────────────────────────────────────────
echo "==== [2/7] numpy 1.26.4 고정 ===="
pip install -q "numpy==1.26.4"

# ── [3/7] detectron2 ──────────────────────────────────────────────────────────
echo "==== [3/7] detectron2 ===="
INSTALLED=false
for TORCH_TAG in "torch${TORCH_MAJOR}.${TORCH_MINOR}" "torch${TORCH_MAJOR}.$((TORCH_MINOR > 0 ? TORCH_MINOR - 1 : 0))" "torch2.6" "torch2.5" "torch2.4" "torch2.3" "torch2.2" "torch2.1" "torch2.0"; do
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
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="$ARCH" \
    pip install -v --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' \
    > /tmp/d2_build.log 2>&1 || {
        echo "=== detectron2 빌드 실패 ==="
        tail -80 /tmp/d2_build.log
        exit 1
    }
fi

# ── [4/7] mmcv-full 1.5.0 소스 빌드 ─────────────────────────────────────────
echo "==== [4/7] mmcv-full 1.5.0 소스 빌드 ===="
TORCH_INC=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')

# SETUPTOOLS_USE_DISTUTILS=local: Python 3.11/3.12에서 distutils 제거 대응
SETUPTOOLS_USE_DISTUTILS=local \
CXXFLAGS="-I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include -I$TORCH_INC/TH -I$TORCH_INC/THC" \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="$ARCH" \
MMCV_WITH_OPS=1 \
pip install -v --no-build-isolation --no-cache-dir 'mmcv-full==1.5.0' > /tmp/mmcv_build.log 2>&1 || {
    echo "=== mmcv-full 빌드 실패 — C++ 에러 ==="
    grep -n "error:" /tmp/mmcv_build.log | grep -v "subprocess-exited\|Could not\|pip's" | head -50
    echo "--- 전체 로그 마지막 200줄 ---"
    tail -200 /tmp/mmcv_build.log
    exit 1
}

# ── [5/7] 버전 고정 패키지 ───────────────────────────────────────────────────
echo "==== [5/7] 버전 고정 패키지 ===="
# 로컬과 동일 버전; tensorflow는 Python 3.11 미지원(2.8.0 기준) + 추론 미사용 → 제외
pip install -q \
    "timm==0.6.13" \
    "pytorch-lightning==2.0.3" \
    "scikit-learn" \
    "mediapipe" \
    "joblib" \
    "smplx"

# ── [6/7] chumpy + numpy 1.x 호환 패치 ──────────────────────────────────────
echo "==== [6/7] chumpy 설치 및 numpy 호환 패치 ===="
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

echo "  -> numpy/opencv 최종 고정 (hamer/ViTPose 설치 후 numpy가 올라갈 수 있어 강제 재설치)"
pip install --force-reinstall -q "numpy==1.26.4" "opencv-python==4.8.1.78"

# ── 검증 ──────────────────────────────────────────────────────────────────────
echo "==== 검증 ===="
python3 -c "
import torch, detectron2, mmcv, timm
import pytorch_lightning as pl, sklearn, cv2, mediapipe as mp
print(f'python            : $(python3 --version)')
print(f'torch             : {torch.__version__}')
print(f'CUDA available    : {torch.cuda.is_available()}')
print(f'detectron2        : {detectron2.__version__}')
print(f'mmcv              : {mmcv.__version__}')
print(f'timm              : {timm.__version__}')
print(f'pytorch_lightning  : {pl.__version__}')
print(f'opencv            : {cv2.__version__}')
print(f'mediapipe         : {mp.__version__}')
" 2>&1 | grep -v "dlerror\|cudart\|Ignore\|LD_LIBRARY"

echo ""
echo "설치 완료! Colab 런타임을 재시작하세요."
