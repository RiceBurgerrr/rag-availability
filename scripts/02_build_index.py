"""
BEIR NQ → Contriever 임베딩 → FAISS flat 인덱스

주의 — 논문 설정과 반드시 일치시켜야 하는 항목
  1) 재청킹 금지: BEIR passage 1개 = 청크 1개
  2) Contriever는 mean pooling (CLS 아님)
  3) 유사도는 dot product (IndexFlatIP), 정규화하지 않음
  4) title + text 결합 방식은 PoisonedRAG 코드와 대조 확인 필요

사용법:
  python 02_build_index.py \
      --data /workspace/data/nq \
      --out  /workspace/indexes/nq_contriever \
      --batch 256 --fp16
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/contriever"


def mean_pooling(token_embeddings, mask):
    """Contriever 공식 pooling. 모델 카드와 동일해야 함."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def load_corpus(path):
    """corpus.jsonl → (ids, texts). 재청킹하지 않는다."""
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            # BEIR/PoisonedRAG 관행: title과 text를 공백으로 결합
            # !! PoisonedRAG 코드의 실제 결합 방식을 확인 후 필요시 수정 !!
            texts.append(f"{title} {body}".strip() if title else body)
            ids.append(d["_id"])
    return ids, texts


@torch.no_grad()
def encode(texts, tok, model, device, batch_size=256, max_len=256, fp16=True, log_every=200):
    out = np.empty((len(texts), model.config.hidden_size),
                   dtype=np.float16 if fp16 else np.float32)
    t0 = time.time()
    for bi, start in enumerate(range(0, len(texts), batch_size)):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            hs = model(**enc).last_hidden_state
        vec = mean_pooling(hs.float(), enc["attention_mask"]).cpu().numpy()
        out[start:start + len(chunk)] = vec.astype(out.dtype)

        if bi % log_every == 0:
            done = start + len(chunk)
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (len(texts) - done) / max(rate, 1e-9)
            print(f"  {done:,}/{len(texts):,}  {rate:.0f}/s  ETA {eta/60:.1f}분", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BEIR nq 디렉토리")
    ap.add_argument("--out", required=True, help="인덱스 출력 디렉토리")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="벡터를 fp16으로 저장 (용량 절반)")
    ap.add_argument("--limit", type=int, default=0, help="디버그용. 0이면 전체")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  fp16={args.fp16}")

    print("[1/4] corpus 로드")
    ids, texts = load_corpus(os.path.join(args.data, "corpus.jsonl"))
    if args.limit:
        ids, texts = ids[:args.limit], texts[:args.limit]
        print(f"  !! 디버그 모드: {args.limit:,}개만 사용. 논문 대조 불가 !!")
    print(f"  passages: {len(texts):,}")

    print("[2/4] Contriever 로드")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("[3/4] 임베딩")
    emb = encode(texts, tok, model, device,
                 batch_size=args.batch, max_len=args.max_len, fp16=args.fp16)
    print(f"  shape={emb.shape}  dtype={emb.dtype}  "
          f"{emb.nbytes / 1e9:.1f}GB")

    print("[4/4] FAISS flat 인덱스 구축")
    import faiss
    dim = emb.shape[1]
    # dot product = IndexFlatIP. 정규화하지 않는다 (MutedRAG 기본 설정)
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, os.path.join(args.out, "faiss.index"))

    np.save(os.path.join(args.out, "embeddings.npy"), emb)
    with open(os.path.join(args.out, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "pooling": "mean",
            "similarity": "dot_product",
            "max_len": args.max_len,
            "n_passages": len(ids),
            "dim": dim,
            "fp16_storage": bool(args.fp16),
            "rechunked": False,
        }, f, indent=2)

    print(f"\n완료: {args.out}")
    print("  faiss.index / embeddings.npy / doc_ids.json / meta.json")


if __name__ == "__main__":
    main()