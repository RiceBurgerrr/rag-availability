#!/usr/bin/env bash
# 파드 시작 시마다 실행. Network Volume(/workspace)에 두고 재사용.
#   bash /workspace/00_setup_pod.sh
#
# Container Disk의 pip 패키지는 파드 삭제 시 사라지므로 매번 필요하다.
# 소요 2~3분.

set -euo pipefail

echo "=== [1/4] 캐시 경로를 Network Volume으로 고정 ==="
# 이걸 안 하면 파드마다 모델 16GB를 새로 받는다
export HF_HOME=/workspace/models
export HF_HUB_CACHE=/workspace/models/hub
export TORCH_HOME=/workspace/models/torch
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"

# 다음 접속에도 적용되도록 기록
grep -q "HF_HOME=/workspace/models" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'EOF'

# --- RAG availability 실험 환경 ---
export HF_HOME=/workspace/models
export HF_HUB_CACHE=/workspace/models/hub
export TORCH_HOME=/workspace/models/torch
export PYTHONUNBUFFERED=1
cd /workspace
EOF

echo "=== [2/4] 시스템 패키지 ==="
apt-get update -qq
apt-get install -y -qq tmux htop wget unzip git ncdu >/dev/null

echo "=== [3/4] 파이썬 패키지 ==="
pip install -q --upgrade pip
pip install -q \
    transformers \
    accelerate \
    sentencepiece \
    faiss-cpu \
    numpy \
    tqdm \
    orjson
# faiss-gpu는 CUDA 버전 충돌이 잦다. flat 인덱스 268만 개 검색은
# CPU에서 질의당 수십 ms라 faiss-cpu로 충분하다.

echo "=== [4/4] 확인 ==="
python3 - <<'PY'
import torch, faiss, transformers, os
print(f"  torch        {torch.__version__}")
print(f"  cuda         {torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM         {total:.0f}GB")
print(f"  faiss        {faiss.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  HF_HOME      {os.environ.get('HF_HOME')}")
PY

echo
echo "완료. 'source ~/.bashrc' 또는 재접속 후 작업 시작."
echo "장시간 작업은 반드시 tmux 안에서 실행할 것:  tmux new -s exp"