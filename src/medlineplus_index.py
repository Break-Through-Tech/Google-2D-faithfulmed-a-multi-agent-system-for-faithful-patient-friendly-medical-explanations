"""
FaithfulMed - MedlinePlus vector index (ChromaDB)

One vector per MedlinePlus health topic. The Simplifier agent calls Retriever.search()
to get plain-language definitions for a medical term.

Put this file in src/ and these three files in data/cleaned-medlineplus/ :
    medlineplus_topics.csv      (Joao's EDA output)
    medlineplus_synonyms.csv    (Joao's EDA output)
    medlineplus_freeze.txt      (records which NLM release the data came from)

Setup:
    pip install chromadb google-genai pandas numpy
    export GEMINI_API_KEY=...

Usage (run from the repo root):
    python src/medlineplus_index.py build       # embed (resumable) + load ChromaDB
    python src/medlineplus_index.py validate    # integrity checks + sample lookups
    python src/medlineplus_index.py query "high blood pressure"

From another agent / notebook:
    from medlineplus_index import Retriever
    hits = Retriever().search("myocardial infarction", k=3)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cleaned-medlineplus"
TOPICS_CSV = DATA_DIR / "medlineplus_topics.csv"
SYNONYMS_CSV = DATA_DIR / "medlineplus_synonyms.csv"
FREEZE_TXT = DATA_DIR / "medlineplus_freeze.txt"
EMBED_CACHE = ROOT / "data" / "embeddings" / "medlineplus_topic_embeddings.npz"
CHROMA_DIR = ROOT / "chroma"
COLLECTION = "medlineplus_topics"

EMBED_MODEL = "gemini-embedding-001"
DOC_TASK = "RETRIEVAL_DOCUMENT"   # used when embedding glossary entries
QUERY_TASK = "RETRIEVAL_QUERY"    # used when embedding a lookup
BATCH_SIZE = 10
MAX_RETRIES = 8


# ----------------------------------------------------------------------------- embeddings
_client = None


def gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a list of texts with Gemini. Waits and retries on rate limits."""
    global _client
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY from the environment

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError(
                    "Gemini DAILY quota used up - waiting will not help. Progress is saved. "
                    "Run 'build' again after the quota resets (see the retry time in the "
                    "error above), or use an API key from a different Google project."
                ) from e
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise
            wait = min(60 * (2 ** attempt), 300)
            print(f"Rate limit reached. Waiting {wait}s ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)


# ----------------------------------------------------------------------------- data loading
def load_data():
    topics = pd.read_csv(TOPICS_CSV)
    synonyms = pd.read_csv(SYNONYMS_CSV)
    freeze = {}
    for line in FREEZE_TXT.read_text().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            freeze[key.strip()] = val.strip()
    return topics, synonyms, freeze


def topic_text(row) -> str:
    """The text that gets embedded (same format as the team's embed.py)."""
    return f"Title: {row.title}\n\nSummary:\n{row.summary}".strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_aliases(topics: pd.DataFrame, synonyms: pd.DataFrame) -> dict[str, list[str]]:
    """topic url -> alternate names (also-called + see-reference), titles excluded.

    Synonym rows pointing at a topic that has no summary (e.g. 'Hernia' in the
    2026-10-13 release) are dropped, because there is nothing to retrieve for them.
    """
    synonyms = synonyms[synonyms.canonical_url.isin(set(topics.url))]
    synonyms = synonyms[synonyms.kind != "title"]
    aliases: dict[str, list[str]] = {}
    for row in synonyms.itertuples():
        aliases.setdefault(row.canonical_url, []).append(str(row.surface))
    return aliases


# ----------------------------------------------------------------------------- embedding cache
def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    """Saved embeddings, keyed by (topic_id, hash of the text that was embedded)."""
    if not EMBED_CACHE.exists():
        return {}
    z = np.load(EMBED_CACHE, allow_pickle=False)
    if str(z["model"]) != EMBED_MODEL or str(z["task_type"]) != DOC_TASK:
        print("Cached embeddings were made with different settings - ignoring them.")
        return {}
    return {
        (int(i), str(h)): v
        for i, h, v in zip(z["ids"], z["hashes"], z["vectors"])
    }


def save_cache(entries: dict) -> None:
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    keys = list(entries)
    tmp = EMBED_CACHE.with_name(EMBED_CACHE.stem + ".tmp.npz")
    np.savez(
        tmp,
        ids=np.array([k[0] for k in keys], dtype=np.int64),
        hashes=np.array([k[1] for k in keys]),
        vectors=np.array([entries[k] for k in keys], dtype=np.float32),
        model=np.array(EMBED_MODEL),
        task_type=np.array(DOC_TASK),
    )
    os.replace(tmp, EMBED_CACHE)


def embed_topics(topics: pd.DataFrame, embed_fn=None) -> dict[int, np.ndarray]:
    """One vector per topic. Resumable: finished topics are cached, and a topic is
    re-embedded automatically if its text changed since the cache was written."""
    embed_fn = embed_fn or gemini_embed
    texts = {int(r.topic_id): topic_text(r) for r in topics.itertuples()}
    key = {tid: (tid, _hash(t)) for tid, t in texts.items()}

    cache = load_cache()
    have = {k: cache[k] for k in key.values() if k in cache}
    todo = [tid for tid in texts if key[tid] not in have]
    print(f"Topics: {len(texts)} | already embedded: {len(have)} | to embed: {len(todo)}")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        vecs = embed_fn([texts[t] for t in batch], DOC_TASK)
        if len(vecs) != len(batch):
            raise ValueError("Embedding count does not match batch size")
        for tid, v in zip(batch, vecs):
            have[key[tid]] = np.asarray(v, dtype=np.float32)
        save_cache(have)  # progress is saved after every batch
        print(f"  embedded {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")

    return {tid: have[key[tid]] for tid in texts}


# ----------------------------------------------------------------------------- build
def build_index(embed_fn=None):
    """Rebuild the ChromaDB collection from scratch (cheap, embeddings are cached)."""
    import chromadb

    topics, synonyms, freeze = load_data()
    vectors = embed_topics(topics, embed_fn)
    aliases = build_aliases(topics, synonyms)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001  (collection did not exist yet)
        pass
    col = client.create_collection(
        COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "source": "MedlinePlus health topics (English)",
            "release_date": freeze.get("release_date", ""),
            "source_sha256": freeze.get("sha256", ""),
            "embedding_model": EMBED_MODEL,
            "doc_task_type": DOC_TASK,
            "query_task_type": QUERY_TASK,
        },
    )

    ids, embs, docs, metas = [], [], [], []
    for r in topics.itertuples():
        tid = int(r.topic_id)
        meta = {
            "topic_id": tid,
            "title": str(r.title),
            "url": str(r.url),
            "n_words": int(r.n_words),
            "also_known_as": json.dumps(aliases.get(r.url, [])),
        }
        if pd.notna(r.fk_grade):
            meta["fk_grade"] = float(r.fk_grade)
        ids.append(str(tid))
        embs.append(vectors[tid].tolist())
        docs.append(str(r.summary))  # the document is the clean lay-language summary
        metas.append(meta)

    for s in range(0, len(ids), 500):
        col.upsert(ids=ids[s:s + 500], embeddings=embs[s:s + 500],
                   documents=docs[s:s + 500], metadatas=metas[s:s + 500])

    print(f"Index built: {col.count()} topics in '{COLLECTION}' at {CHROMA_DIR}")
    return col


# ----------------------------------------------------------------------------- query
class Retriever:
    """What the Simplifier calls.

    search(term) returns a list of dicts (best first), each with:
        topic_id, title, url, definition, score, match_type, also_known_as
    - match_type "synonym": the text exactly matched a topic title or alternate name
      (e.g. "zits" -> Acne). score is 1.0.
    - match_type "vector": nearest topics by meaning, NOT a guaranteed definition of the
      query. score is cosine similarity (higher = closer). With gemini-embedding-001 the
      scores of good and poor matches overlap (about 0.66-0.77 in our checks), so a
      min_score cutoff cannot reliably tell them apart. A drug name like "metformin"
      returns the related topic "Diabetes Medicines", not a definition of metformin.
      Treat "synonym" hits as definitions and "vector" hits as related background only.
    Query with a single term or short phrase, not a whole paragraph.
    """

    def __init__(self, embed_fn=None, collection=None):
        if collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(COLLECTION)
        self.col = collection
        self.embed_fn = embed_fn or gemini_embed

        # Exact-match lookup (titles + alternate names), built from the index itself
        got = self.col.get(include=["metadatas"])
        self.surface_to_id: dict[str, str] = {}
        for tid, meta in zip(got["ids"], got["metadatas"]):
            names = [meta["title"]] + json.loads(meta.get("also_known_as", "[]"))
            for name in names:
                self.surface_to_id.setdefault(norm(name), tid)

    @staticmethod
    def _format(doc, meta, score, match_type):
        return {
            "topic_id": meta["topic_id"],
            "title": meta["title"],
            "url": meta["url"],
            "definition": doc,
            "score": round(float(score), 4),
            "match_type": match_type,
            "also_known_as": json.loads(meta.get("also_known_as", "[]")),
        }

    def search(self, query: str, k: int = 3, min_score: float = 0.0,
               use_synonyms: bool = True) -> list[dict]:
        if use_synonyms and norm(query) in self.surface_to_id:
            tid = self.surface_to_id[norm(query)]
            got = self.col.get(ids=[tid], include=["documents", "metadatas"])
            return [self._format(got["documents"][0], got["metadatas"][0], 1.0, "synonym")]

        qvec = [float(x) for x in self.embed_fn([query], QUERY_TASK)[0]]
        res = self.col.query(query_embeddings=[qvec], n_results=k,
                             include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - dist  # Chroma cosine distance = 1 - cosine similarity
            if score >= min_score:
                out.append(self._format(doc, meta, score, "vector"))
        return out


# ----------------------------------------------------------------------------- validate
SEMANTIC_PROBES = [  # (lookup, title we expect in the top 3)
    ("a blood test that shows average blood sugar over three months", "A1C"),
    ("swelling in the legs caused by fluid buildup", "Edema"),
    ("wheezing and trouble breathing from inflamed airways", "Asthma"),
    ("pimples on the face from clogged pores", "Acne"),
]
NOT_IN_GLOSSARY = ["metformin", "warfarin", "troponin", "stent", "beta-blocker"]


def validate(embed_fn=None, collection=None) -> bool:
    """Integrity checks (no API needed) + sample lookups (API needed)."""
    topics, _, freeze = load_data()
    if collection is None:
        import chromadb
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    col = collection
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    print(f"Collection metadata: {col.metadata}")
    print(f"CSV release: {freeze.get('release_date')}\n")

    got = col.get(include=["embeddings", "metadatas"])
    check("entry count matches CSV", col.count() == len(topics),
          f"(index {col.count()}, csv {len(topics)})")
    check("every topic_id present", {str(t) for t in topics.topic_id} == set(got["ids"]))

    mat = np.asarray(got["embeddings"], dtype=np.float32)
    check("no NaN values", not np.isnan(mat).any())
    check("no empty (all-zero) vectors", bool((np.abs(mat).sum(axis=1) > 0).all()))
    check("all vectors same dimension", mat.ndim == 2, f"(dim = {mat.shape[-1]})")
    check("no duplicate vectors", len({row.tobytes() for row in mat}) == len(mat))
    check("metadata complete", all(m.get("title") and m.get("url") for m in got["metadatas"]))

    rng = np.random.default_rng(0)
    sample = rng.choice(len(mat), size=min(25, len(mat)), replace=False)
    self_hits = 0
    for i in sample:
        res = col.query(query_embeddings=[mat[i].tolist()], n_results=1)
        self_hits += res["ids"][0][0] == got["ids"][i]
    check("a topic's own vector retrieves itself", self_hits == len(sample),
          f"({self_hits}/{len(sample)})")

    print("\n--- Semantic lookups (these call the Gemini API; 'look' = check by eye) ---")
    r = Retriever(embed_fn=embed_fn, collection=col)
    for query, expected in SEMANTIC_PROBES:
        hits = r.search(query, k=3, use_synonyms=False)
        titles = [h["title"] for h in hits]
        flag = "ok  " if expected in titles else "look"
        print(f"[{flag}] '{query}' -> {[(h['title'], h['score']) for h in hits]}")

    print("\n--- Terms with no glossary entry: shows what comes back (scores overlap with real matches) ---")
    for term in NOT_IN_GLOSSARY:
        hits = r.search(term, k=1, use_synonyms=False)
        top = hits[0] if hits else None
        print(f"  {term:14s} -> {(top['title'], top['score']) if top else None}")

    print("\n--- Exact-match (synonym) lookups ---")
    for term in ["myocardial infarction", "zits", "hypertension"]:
        hits = r.search(term)
        print(f"  {term:22s} -> {hits[0]['title'] if hits else None} "
              f"({hits[0]['match_type'] if hits else '-'})")

    print("\nStructural checks:", "ALL PASSED" if ok else "SOME FAILED")
    return ok


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "build":
        build_index()
    elif cmd == "validate":
        return 0 if validate() else 1
    elif cmd == "query" and len(argv) > 2:
        for h in Retriever().search(" ".join(argv[2:]), k=3):
            print(f"{h['score']:.3f}  [{h['match_type']}]  {h['title']}  {h['url']}")
            print(f"       {h['definition'][:160]}...")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
