# rag-availability

# NQ 실험 파이프라인 — 시작 코드

> 실험설계서 §8 수행 순서의 ①~⑤에 해당하는 부분. RunPod에서 실행.

## 왜 기존 레포를 그대로 쓰지 않는가

| 버릴 것 | 사유 |
|---|---|
| `src/query_app.py` (1,711줄) | `block_type`(table_row / clause_section), 구조화 라우팅, 문맥 확장 등 대부분이 **한국어 docx 전용**. BEIR passage는 평평한 텍스트라 해당 없음 |
| `src/index_builder.py`, `retrievers.py` | BM25 중심 + 500자 청킹 전제. BEIR는 재청킹 금지 |
| `capstone_html/`, `backup/`, `pipeline_*/` | 서비스 데모용 |

| 이식할 것 | 사유 |
|---|---|
| `detector/scoring.py`, `risk.py`의 **점수 계산식** | 연구 산출물. 다시 짜면 캡스톤 결과와 연속성이 끊김 |
| `detector/runtime.py`의 상호작용 항 공식 | `combined_risk`, `interaction_boost` 계산은 그대로 유지 |
| 위험도 가중치·임계값 | 재보정의 출발점 |
| 재검색 루프 **로직** | 코드가 아니라 순서(제외 누적 → 재검색 → 재검사) |

`detector/patterns.py`는 한국어 기반이므로 영어로 재작성해야 한다.

## 실험 하네스 설계 원칙

서비스 코드와 다르게 설계한다.

**① 단계 분리 + 중간 결과 영속화**

```
검색  → runs/<exp>/retrieval.jsonl
생성  → runs/<exp>/generation.jsonl
판정  → runs/<exp>/judgment.jsonl
```

judge 프롬프트를 수정했다고 24,000회 생성을 다시 돌리면 안 된다. 분리해두면 판정 단계만 재실행한다.

**② 결정론** — seed 고정, `temperature=0`. 또는 3회 반복 후 평균±표준편차.

**③ 캐싱** — 동일한 (질의, 컨텍스트) 조합의 생성 결과 재사용. 조건 간 중복이 많다.

**④ 전량 로깅** — 검색된 청크 ID, 신호별 점수, 제외 사유를 모두 남긴다. 결과가 이상할 때 원인 추적이 불가능해진다.

## 디렉토리 구조 (제안)

```
rag-availability/
├── data/            # BEIR NQ (Network Volume)
├── index/           # 02_build_index.py, 03_retriever.py
├── attacks/         # 04_attacks.py, bbo/ (Jamming 공식 코드)
├── defense/         # signals.py(이식), decide.py, recover.py
├── eval/            # judge.py, metrics.py
├── runs/            # 실행 스크립트 + 결과 JSONL
└── configs/
```

## 실행 순서

```bash
# 1. 데이터 다운로드 (Network Volume)
bash 01_download_nq.sh /workspace/data

# 2. 인덱스 구축 (~1.5h, 1회성)
python 02_build_index.py \
    --data /workspace/data/nq \
    --out  /workspace/indexes/nq_contriever \
    --batch 256 --fp16

# 3. 검색 동작 확인
python 03_retriever.py \
    --index  /workspace/indexes/nq_contriever \
    --corpus /workspace/data/nq/corpus.jsonl

# 4. (A0 실행 후) 공격 문서 생성
python 04_attacks.py \
    --queries    /workspace/runs/a0/target_queries.jsonl \
    --jailbreaks /workspace/data/jailbreakbench.txt \
    --out        /workspace/attacks/attacks.jsonl
```

**디버그 시에는 `--limit 50000`으로 소규모 인덱스를 먼저 만들어 파이프라인을 검증한다.** 단 그 결과는 논문 대조에 사용할 수 없다(코퍼스 축소는 IR을 인위적으로 상승시킴).

## 논문 설정과 대조 확인할 항목

착수 전 PoisonedRAG 공식 코드(`github.com/sleeepeer/PoisonedRAG`)와 대조한다. 어긋나면 ASR이 논문과 달라지고 원인 추적이 어렵다.

| 항목 | 본 코드 | 확인 필요 |
|---|---|---|
| pooling | mean | Contriever 모델 카드와 동일 ✅ |
| 유사도 | dot product (`IndexFlatIP`) | 정규화하지 않음 ✅ |
| top-k | 5 | MutedRAG 기본값 ✅ |
| **title + text 결합** | `f"{title} {body}"` | **PoisonedRAG 실제 방식 확인 필요** ⚠️ |
| max_length | 256 | PoisonedRAG 설정 확인 필요 ⚠️ |
| 재청킹 | 하지 않음 | ✅ |

## 다음 단계

- `05_generate.py` — LLM 생성 (vLLM 또는 Ollama)
- `06_judge.py` — 2단 판정 (패턴 매칭 + LLM judge)
- `07_metrics.py` — IR / I-ASR / ASR 집계
- `defense/` — detector 이식 + 재검색 루프

이 네 개는 A0 결과를 보고 나서 구조를 확정하는 게 낫다.
