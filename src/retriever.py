"""
검색기 + 공격 문서 주입

핵심 포인트
  - 공격 문서를 넣는다고 268만 개를 다시 임베딩하지 않는다.
    공격 문서만 인코딩해서 기존 인덱스에 add 한다 (수 초).
  - 조건별로 인덱스를 분리하려면 base 인덱스를 복사한 뒤 각각 add.
  - 제외 목록(exclusion) 기반 재검색을 지원한다. 방어 로직이 이걸 호출한다.
"""
import json, os, copy
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/contriever"


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


class Retriever:
    def __init__(self, index_dir, corpus_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()

        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        # 본문은 검색 후에만 필요하므로 id → text 맵을 메모리에 올린다.
        # NQ 기준 약 2~3GB. RAM이 부족하면 sqlite로 대체할 것.
        self.texts = {}
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                t = (d.get("title") or "").strip()
                b = (d.get("text") or "").strip()
                self.texts[d["_id"]] = f"{t} {b}".strip() if t else b

    @torch.no_grad()
    def encode(self, texts, max_len=256):
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=max_len, return_tensors="pt").to(self.device)
        hs = self.model(**enc).last_hidden_state
        return mean_pooling(hs.float(), enc["attention_mask"]).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # 공격 문서 주입
    # ------------------------------------------------------------------
    def inject(self, attack_docs):
        """
        attack_docs: [{"_id": "atk_q45", "text": "..."}]
        기존 인덱스에 추가. 268만 개 재임베딩 불필요.
        """
        if not attack_docs:
            return
        vecs = self.encode([d["text"] for d in attack_docs])
        self.index.add(vecs)
        for d in attack_docs:
            self.doc_ids.append(d["_id"])
            self.texts[d["_id"]] = d["text"]
        print(f"  주입 완료: {len(attack_docs)}개 (총 {self.index.ntotal:,})")

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------
    def search(self, query, top_k=5, exclude_ids=None, exclude_sources=None):
        """
        exclude_ids     : 제외할 청크 ID 집합 (방어의 재검색에서 사용)
        exclude_sources : 제외할 출처 집합

        제외분을 채우기 위해 여유분을 더 가져온 뒤 필터링한다.
        """
        exclude_ids = set(exclude_ids or [])
        exclude_sources = set(exclude_sources or [])
        fetch = top_k + len(exclude_ids) + 20  # 여유분

        qv = self.encode([query])
        scores, idxs = self.index.search(qv, min(fetch, self.index.ntotal))

        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            cid = self.doc_ids[i]
            if cid in exclude_ids:
                continue
            src = cid.split("#")[0]
            if src in exclude_sources:
                continue
            out.append({
                "chunk_id": cid,
                "source": src,
                "text": self.texts.get(cid, ""),
                "score": float(score),
            })
            if len(out) >= top_k:
                break
        return out


def load_queries(path, ids=None):
    """queries.jsonl → [{"qid","text"}]"""
    want = set(ids) if ids else None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if want and d["_id"] not in want:
                continue
            out.append({"qid": d["_id"], "text": d["text"]})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--query", default="who owns the four seasons hotel in las vegas")
    args = ap.parse_args()

    r = Retriever(args.index, args.corpus)
    for h in r.search(args.query, top_k=5):
        print(f"[{h['score']:.3f}] {h['chunk_id']}  {h['text'][:120]}...")