#!/usr/bin/env bash
# BEIR NQ 다운로드 — RunPod Network Volume에 저장
# 사용법: bash 01_download_nq.sh /workspace/data

set -euo pipefail
DATA_DIR="${1:-/workspace/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[1/3] BEIR NQ 다운로드"
if [ ! -f nq.zip ]; then
  wget -c https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
fi

echo "[2/3] 압축 해제"
if [ ! -d nq ]; then
  unzip -q nq.zip
fi

echo "[3/3] 검증"
python3 - <<'PY'
import json, os

base = "nq"
c = os.path.join(base, "corpus.jsonl")
q = os.path.join(base, "queries.jsonl")

def count(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)

def first(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())

n_corpus = count(c)
n_query  = count(q)

print(f"  corpus  : {n_corpus:,}   (MutedRAG Table 1 = 2,681,468)")
print(f"  queries : {n_query:,}   (MutedRAG Table 1 = 3,452)")
print(f"  corpus 예시: {first(c)}")
print(f"  query  예시: {first(q)}")

# 논문 수치와 어긋나면 다른 배포판이므로 즉시 중단
assert n_corpus == 2_681_468, f"코퍼스 수 불일치: {n_corpus}"
print("\n  OK: 논문과 동일한 배포판")
PY

echo
echo "완료. 경로: $DATA_DIR/nq"