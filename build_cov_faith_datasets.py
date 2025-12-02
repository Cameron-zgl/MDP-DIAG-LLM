#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDP—Coverage & Faithfulness Dataset Builder (WHO topics CSV + pathology textbook)

What this script does
---------------------
1) Loads sources:
   - WHO CSV with columns: topic, url, text, source
   - Pathology textbook plain text file
2) Chunks documents and builds a simple TF‑IDF retriever (optionally hybrid w/ BM25 if rank_bm25 installed).
3) Uses DeepSeek (optional) to generate QA pairs per chunk with recorded provenance.
4) Constructs two task datasets:
   A) Coverage / Ground‑truth Recall: question, answer, ground‑truth (doc_id, chunk_id), top‑K retrieval, Hit@K, ContextRecall@K
   B) Context Consistency / Faithfulness: positive (faithful) and negative (unfaithful) answer variants + optional LLM NLI faithfulness score.
5) Saves artifacts to CSV (and JSONL for retrieval lists) under an output folder.

Quick start
-----------
python build_cov_faith_datasets.py \
  --who_csv /home/gulizhu/MDP/combined_health_topics_with_source.csv \
  --textbook_path /home/gulizhu/MDP/textbook_pathology.txt \
  --out_dir /home/gulizhu/MDP/benchmark_data/coverage_faithfulness \
  --k 10 --max_qas_per_chunk 2 --sample_chunks 400 \
  --use_deepseek

Environment
-----------
export DEEPSEEK_API_KEY=sk-...   # required if --use_deepseek or --faith_nli

Notes
-----
- DeepSeek calls are optional; if skipped, rule‑based QA seeds + paraphrases are used.
- You can later plug in your own retrievers; we record doc_id/chunk_id so it’s drop‑in for evaluation.
"""

from __future__ import annotations
import os, re, json, argparse, random, hashlib, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# === Retrieval: TF‑IDF (sklearn) and optional BM25 ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi  # pip install rank_bm25
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# === LLM client (DeepSeek) optional ===
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Utilities
# -----------------------

def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:10]


def normalize_ws(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


# -----------------------
# Corpus & Chunking
# -----------------------
@dataclass
class Doc:
    doc_id: str
    source: str  # e.g., WHO, TEXTBOOK
    title: str
    url: Optional[str]
    text: str


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int


def load_who_csv(path: str) -> List[Doc]:
    df = pd.read_csv(path)
    docs = []
    for _, r in df.iterrows():
        title = str(r.get('topic', ''))
        url = str(r.get('url', '')) if not pd.isna(r.get('url')) else None
        text = str(r.get('text', ''))
        source = str(r.get('source', 'WHO'))
        doc_id = f"WHO::{md5(title + (url or '') )}"
        docs.append(Doc(doc_id=doc_id, source=source, title=title, url=url, text=text))
    return docs


def load_textbook(path: str) -> List[Doc]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    # Split by very likely headings or long blank lines
    parts = re.split(r"\n\s*\n+", raw)
    docs = []
    for i, part in enumerate(parts):
        title = f"Pathology Section {i+1}"
        doc_id = f"TEXTBOOK::{i+1:04d}"
        docs.append(Doc(doc_id=doc_id, source='TEXTBOOK', title=title, url=None, text=part))
    return docs


def greedy_chunk(text: str, max_chars: int = 1200, overlap: int = 120) -> List[Tuple[str, int, int]]:
    """Return list of (chunk_text, start, end)."""
    text = normalize_ws(text)
    n = len(text)
    chunks = []
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        # try to cut at sentence end
        cut = text.rfind('. ', i, j)
        if cut == -1 or cut < i + 200:
            cut = j
        else:
            cut = cut + 1  # include dot
        chunk = text[i:cut]
        chunks.append((chunk, i, cut))
        i = max(cut - overlap, i + 1)
    return chunks


def build_corpus_chunks(docs: List[Doc], max_chars=1200, overlap=120) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        for idx, (ctext, s, e) in enumerate(greedy_chunk(d.text, max_chars, overlap)):
            chunk_id = f"{d.doc_id}::CH{idx:04d}"
            chunks.append(Chunk(doc_id=d.doc_id, chunk_id=chunk_id, text=ctext, start_char=s, end_char=e))
    return chunks


# -----------------------
# Retriever
# -----------------------
class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.ids = [c.chunk_id for c in chunks]
        self.vectorizer = TfidfVectorizer(max_features=60000, ngram_range=(1,2))
        self.tf_matrix = self.vectorizer.fit_transform(self.texts)
        if HAS_BM25:
            self.bm25 = BM25Okapi([t.split() for t in self.texts])
        else:
            self.bm25 = None

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = [query]
        tf_q = self.vectorizer.transform(q)
        tf_scores = cosine_similarity(tf_q, self.tf_matrix).ravel()
        tf_rank = list(enumerate(tf_scores))
        tf_rank.sort(key=lambda x: x[1], reverse=True)
        if self.bm25 is not None:
            bm_scores = self.bm25.get_scores(query.split())
            # simple hybrid: normalized sum
            bm_scores = (bm_scores - np.min(bm_scores)) / (np.max(bm_scores) - np.min(bm_scores) + 1e-9)
            tf_scores_n = (tf_scores - np.min(tf_scores)) / (np.max(tf_scores) - np.min(tf_scores) + 1e-9)
            hybrid = tf_scores_n * 0.5 + bm_scores * 0.5
            rank = list(enumerate(hybrid))
            rank.sort(key=lambda x: x[1], reverse=True)
        else:
            rank = tf_rank
        top = rank[:k]
        return [(self.ids[i], float(s)) for i, s in top]


# -----------------------
# QA generation (optional DeepSeek)
# -----------------------
def get_ds_client() -> Optional[OpenAI]:
    if not HAS_OPENAI:
        return None
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_make_qas(client: OpenAI, text: str, n: int = 2) -> List[Tuple[str,str]]:
    prompt = (
        "You are a biomedical editor. Read the passage and create concise, factual QA pairs.\n"
        "Rules: Focus on atomic facts that can be answered directly from the passage. Avoid multi-sentence answers.\n"
        f"PASSAGE:\n{text}\n\nReturn a JSON list of objects with 'q' and 'a'."
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2, max_tokens=600
        )
        content = resp.choices[0].message.content
        # Extract JSON
        m = re.search(r"\[.*\]", content, flags=re.S)
        items = json.loads(m.group(0)) if m else []
        out = []
        for it in items[:n]:
            q = normalize_ws(it.get('q',''))
            a = normalize_ws(it.get('a',''))
            if q and a:
                out.append((q,a))
        return out
    except Exception:
        return []


def rule_make_qas(text: str, n: int = 2) -> List[Tuple[str,str]]:
    # Fallback: pick sentences and form definition / list questions
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s for s in sents if 40 < len(s) < 300]
    out: List[Tuple[str,str]] = []
    for s in sents[: n*3]:
        # define question with the first NP
        q = "What does the passage say about this topic?"
        a = normalize_ws(s)
        out.append((q, a))
        if len(out) >= n:
            break
    return out


# -----------------------
# Faithfulness synthesis helpers
# -----------------------
NEG_SWAP_TABLE = {
    # domain-agnostic polarity
    r"\bincrease(s|d)?\b": "decrease",
    r"\bdecrease(s|d)?\b": "increase",
    r"\bhigher\b": "lower",
    r"\blower\b": "higher",
    # oncology-ish example
    r"\badenocarcinoma\b": "squamous cell carcinoma",
    r"\bsquamous( cell)? carcinoma\b": "adenocarcinoma",
}


def make_unfaithful(answer: str) -> Optional[str]:
    cand = answer
    flips = 0
    for pat, repl in NEG_SWAP_TABLE.items():
        if re.search(pat, cand, flags=re.I):
            cand = re.sub(pat, repl, cand, flags=re.I)
            flips += 1
    if flips == 0:
        return None
    return cand


def llm_faithfulness_score(client: OpenAI, answer: str, evidence: str) -> float:
    prompt = (
        "You are a strict NLI judge. Score if the ANSWER is entailed by the EVIDENCE from 0.0 to 1.0.\n"
        "Return only a JSON object: {\"score\": number}.\n\n"
        f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}\n"
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=60
        )
        content = resp.choices[0].message.content
        m = re.search(r"\{.*\}", content, flags=re.S)
        obj = json.loads(m.group(0)) if m else {"score": 0.0}
        score = float(obj.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


# -----------------------
# Builders
# -----------------------

def build_coverage_dataset(
    chunks: List[Chunk], retriever: HybridRetriever, k: int, max_qas_per_chunk: int,
    sample_chunks: int, use_deepseek: bool
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str,float]]]]:
    client = get_ds_client() if use_deepseek else None
    rows = []
    retrieval_lists: Dict[str, List[Tuple[str,float]]] = {}

    # sample chunks for diversity
    pool = random.sample(list(range(len(chunks))), min(sample_chunks, len(chunks)))
    for idx in pool:
        ch = chunks[idx]
        # make QAs
        qas = llm_make_qas(client, ch.text, n=max_qas_per_chunk) if client else rule_make_qas(ch.text, n=max_qas_per_chunk)
        for (q, a) in qas:
            qid = f"Q::{md5(q + ch.chunk_id)}"
            # search
            top = retriever.search(q, k=k)
            retrieval_lists[qid] = top
            top_ids = [cid for cid, _ in top]
            hit_doc = any(cid.startswith(ch.doc_id) for cid in top_ids)
            hit_chunk = ch.chunk_id in top_ids
            # lightweight token recall vs contexts
            joined_ctx = " \n\n".join([retriever.chunks[retriever.ids.index(cid)].text for cid in top_ids if cid in retriever.ids])
            ans_tokens = set([t.lower() for t in re.findall(r"\b\w+\b", a) if len(t) > 3])
            ctx_tokens = set([t.lower() for t in re.findall(r"\b\w+\b", joined_ctx) if len(t) > 3])
            overlap = ans_tokens.intersection(ctx_tokens)
            context_recall = len(overlap) / (len(ans_tokens) + 1e-9)

            rows.append({
                'qid': qid,
                'question': q,
                'answer': a,
                'gt_doc_id': ch.doc_id,
                'gt_chunk_id': ch.chunk_id,
                'hit_doc@K': int(hit_doc),
                'hit_chunk@K': int(hit_chunk),
                'context_recall@K': round(float(context_recall), 4),
            })

    df = pd.DataFrame(rows)
    return df, retrieval_lists


def build_faithfulness_dataset(
    cov_df: pd.DataFrame, chunks_by_id: Dict[str, Chunk], retrieval_lists: Dict[str, List[Tuple[str,float]]],
    retriever: HybridRetriever, use_deepseek_nli: bool
) -> pd.DataFrame:
    client = get_ds_client() if use_deepseek_nli else None
    rows = []
    for _, r in cov_df.iterrows():
        qid = r['qid']
        q = r['question']
        a = r['answer']
        gt_chunk_id = r['gt_chunk_id']
        gt_chunk = chunks_by_id.get(gt_chunk_id)
        evidence_text = gt_chunk.text if gt_chunk else ''
        # Positive (faithful) example
        pos_score = llm_faithfulness_score(client, a, evidence_text) if client else 1.0
        rows.append({
            'qid': qid,
            'question': q,
            'answer': a,
            'label_faithful': 1,
            'evidence_chunk_id': gt_chunk_id,
            'faithfulness_score': round(float(pos_score), 3)
        })
        # Negative (unfaithful) example (synthetic swap)
        neg_a = make_unfaithful(a)
        if neg_a:
            neg_score = llm_faithfulness_score(client, neg_a, evidence_text) if client else 0.0
            rows.append({
                'qid': qid,
                'question': q,
                'answer': neg_a,
                'label_faithful': 0,
                'evidence_chunk_id': gt_chunk_id,
                'faithfulness_score': round(float(neg_score), 3)
            })
        # Also evaluate faithfulness w.r.t. retrieved (top‑1) evidence to catch citation‑precision errors
        top = retrieval_lists.get(qid, [])
        if top:
            top1_id = top[0][0]
            top1_chunk = chunks_by_id.get(top1_id)
            if top1_chunk:
                nli_top1 = llm_faithfulness_score(client, a, top1_chunk.text) if client else 1.0 if top1_id==gt_chunk_id else 0.5
                rows.append({
                    'qid': qid,
                    'question': q,
                    'answer': a,
                    'label_faithful': int(top1_id == gt_chunk_id),  # use as proxy for citation precision
                    'evidence_chunk_id': top1_id,
                    'faithfulness_score': round(float(nli_top1), 3),
                    'note': 'top1_evidence_eval'
                })
    return pd.DataFrame(rows)


# -----------------------
# Save / Load helpers
# -----------------------

def save_retrieval_lists(path: str, retrieval_lists: Dict[str, List[Tuple[str,float]]]):
    with open(path, 'w', encoding='utf-8') as f:
        for qid, items in retrieval_lists.items():
            rec = {'qid': qid, 'topk': [{'chunk_id': cid, 'score': s} for cid, s in items]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--who_csv', type=str, required=True)
    ap.add_argument('--textbook_path', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--max_qas_per_chunk', type=int, default=2)
    ap.add_argument('--sample_chunks', type=int, default=300)
    ap.add_argument('--use_deepseek', action='store_true')
    ap.add_argument('--faith_nli', action='store_true', help='use DeepSeek for NLI faithfulness scoring')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load docs
    who_docs = load_who_csv(args.who_csv)
    tb_docs = load_textbook(args.textbook_path)
    all_docs = who_docs + tb_docs

    # 2) Chunk
    chunks = build_corpus_chunks(all_docs, max_chars=1100, overlap=100)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    # 3) Retriever
    retriever = HybridRetriever(chunks)

    # 4) Coverage dataset
    cov_df, retrieval_lists = build_coverage_dataset(
        chunks, retriever, k=args.k, max_qas_per_chunk=args.max_qas_per_chunk,
        sample_chunks=args.sample_chunks, use_deepseek=args.use_deepseek
    )

    cov_csv = out_dir / 'coverage_dataset.csv'
    cov_df.to_csv(cov_csv, index=False)
    save_retrieval_lists(str(out_dir / 'coverage_retrieval_topk.jsonl'), retrieval_lists)

    # 5) Faithfulness dataset (uses coverage set as seed)
    faith_df = build_faithfulness_dataset(cov_df, chunks_by_id, retrieval_lists, retriever, use_deepseek_nli=args.faith_nli)
    faith_csv = out_dir / 'faithfulness_dataset.csv'
    faith_df.to_csv(faith_csv, index=False)

    # 6) Simple aggregate metrics snapshot for the generated coverage set (as a sanity check)
    if len(cov_df):
        snapshot = {
            'num_questions': int(len(cov_df)),
            f'hit_doc@{args.k}': float(cov_df['hit_doc@K'].mean()),
            f'hit_chunk@{args.k}': float(cov_df['hit_chunk@K'].mean()),
            f'avg_context_recall@{args.k}': float(cov_df['context_recall@K'].mean()),
        }
        with open(out_dir / 'coverage_snapshot.json', 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", cov_csv)
    print(" -", out_dir / 'coverage_retrieval_topk.jsonl')
    print(" -", faith_csv)
    print(" -", out_dir / 'coverage_snapshot.json')


if __name__ == '__main__':
    main()
