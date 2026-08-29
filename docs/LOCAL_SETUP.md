# 로컬 세팅 (RunPod 착수 전)

> 대상: RTX 4080 SUPER 16GB / RAM 16GB. **인덱스 구축은 하지 않는다** — RAM 부족이고 RunPod에서 할 일이다.
> 목적: 9월 착수 시 막힐 지점을 미리 뚫고, 미결 안건 하나를 데이터로 해결한다.

---

## 오늘 할 것 — 신호 사전검증

**미결 안건 1순위였던 "PPL 신호를 구현할 것인가"에 답한다.**

서론 기여 3이 *"명령 표면 신호와 문맥 자연성 신호가 서로 다른 공격 변종을 보완적으로 포착한다"* 는 주장 위에 서 있다. 이것이 실제로 성립하는지 측정하지 않으면 기여를 확정할 수 없다.

**인덱스 없이 가능하다.** 문서 텍스트만 있으면 PPL과 Prompt Guard 측정이 된다.

### 준비

```powershell
# 1. 가상환경
python -m venv .venv
.venv\Scripts\activate

# 2. 패키지 (torch는 CUDA 버전에 맞춰 설치)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate sentencepiece

# 3. NQ 다운로드 (4GB, 디스크만 사용)
curl -L -o nq.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
tar -xf nq.zip          # 또는 압축 프로그램으로 해제

# 4. Prompt Guard는 gated 모델이라 로그인 필요
huggingface-cli login   # HF 토큰 입력
```

`meta-llama/Llama-Prompt-Guard-2-86M` 페이지에서 **라이선스 동의를 먼저 눌러야** 다운로드가 된다. 승인이 즉시 나지 않을 수 있으니 오늘 미리 신청해 두는 것이 좋다.

### 실행

```powershell
python 05_signal_probe.py ^
    --corpus  .\nq\corpus.jsonl ^
    --queries .\nq\queries.jsonl ^
    --n 100 ^
    --out .\signal_probe.json
```

Prompt Guard 승인이 안 났으면 PPL만 먼저 본다.

```powershell
python 05_signal_probe.py --corpus .\nq\corpus.jsonl --queries .\nq\queries.jsonl --n 100 --skip-pg
```

몇 분이면 끝난다. GPT-2와 Prompt Guard 모두 1GB 미만이라 4080에 부담이 없다.

### 결과 해석

| PPL AUC | 판정 |
|---|---|
| ≥ 0.90 | 해당 공격은 **PPL 단독으로 분리 가능** |
| 0.70 ~ 0.90 | 보조 신호로는 유효하나 단독은 불안정 |
| ≤ 0.70 | **PPL로는 부족**. 다른 축 필요 |

**예상 결과** (Jamming·TabooRAG 논문 기준)

| 그룹 | PPL | Prompt Guard |
|---|---|---|
| clean | 낮음 (참고: 15.93) | 미탐지 |
| mutedrag | 높음 | 탐지 |
| jamming_ii | **정상 범위** | 탐지 |
| bbo_proxy | **매우 높음** (참고: 290.64) | **미탐지** |

**이 패턴이 재현되면** 신호 상보성 논거가 실증되므로 **PPL 구현을 확정**한다.
**bbo_proxy의 PPL AUC가 낮게 나오면** 프록시가 실제 BBO를 제대로 모사하지 못한 것이므로, 판단을 보류하고 실제 BBO 문서로 재측정한다.

**clean 의 Prompt Guard 탐지율도 반드시 확인한다.** 이 값이 높으면 오탐이 심하다는 뜻이고, TabooRAG 논문이 보고한 *"procedural queries가 instruction으로 오분류되어 정상 문서를 과잉 차단"* 현상이 재현된 것이다. 그 자체가 논문에 쓸 결과다.

---

## 오늘 같이 해두면 좋은 것

### ① JailbreakBench 확보

MutedRAG 공격 문서에 필요하다. 현재 스크립트는 내장 최소표본 4개로 동작하나, **실제 실험에는 JailbreakBench 100개가 필요**하다.

HuggingFace `JailbreakBench/JBB-Behaviors` 에서 받아 한 줄에 하나씩 텍스트 파일로 저장한 뒤 `--jailbreaks` 로 넘긴다.

### ② GitHub 레포 구성

9월에 팀원들이 각자 파드에서 `git pull` 만 하면 되도록 미리 올려둔다.

```
rag-availability/
├── index/      02_build_index.py, 03_retriever.py
├── attacks/    04_attacks.py
├── probe/      05_signal_probe.py
├── scripts/    00_setup_pod.sh, 01_download_nq.sh
└── README.md
```

**모델 가중치와 인덱스는 절대 커밋하지 않는다.** `.gitignore` 에 `*.index`, `*.npy`, `data/`, `models/` 를 넣는다.

### ③ patterns.py 영어 재작성

`detector/patterns.py` 가 한국어 기준이라 NQ에서 작동하지 않는다. 이건 GPU가 필요 없으니 로컬에서 하면 된다.

기존 다섯 신호(명령 패턴 / 프롬프트 표식 / 명령 형식 / 명령형 표현 비율 / 정책 용어 밀도)에 대응하는 영어 패턴을 작성하고, 위 사전검증에서 만든 공격 문서로 발화 여부를 확인한다.

---

## 하지 않을 것

| | 사유 |
|---|---|
| 인덱스 구축 | 벡터 4.1GB(fp16) + 코퍼스 로딩에 RAM 16GB로는 빠듯. RunPod에서 한다 |
| LLM 생성 실험 | Llama-2-7b fp16이 13.5GB로 4080 16GB에 빠듯하고, 양자화하면 가드레일 동작이 바뀌어 결과를 못 쓴다 |
| 코퍼스 축소 실험 | IR이 왜곡되어 논문 대조가 불가능해진다 |

---

## 이 결과가 바꾸는 것

사전검증 결과에 따라 다음이 결정된다.

| 결과 | 조치 |
|---|---|
| 상보성 패턴 재현됨 | **PPL 구현 확정.** 서론 기여 3 그대로 유지 |
| PPL이 어느 공격도 못 잡음 | 기여 3 재작성. 신호 축을 다시 설계 |
| Prompt Guard 오탐 높음 | **베이스라인 논거 강화.** "차단은 정상 문서까지 막는다"를 우리 데이터로 뒷받침 |

측정이 끝나면 설계서 §0-0 신호 커버리지 표와 §10.3을 **우리 숫자로** 채운다.