# rag.py
import os
import json
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import nltk

# Download NLTK packages (first time only)
nltk_packages = ["punkt", "stopwords"]
for p in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{p}")
    except Exception:
        nltk.download(p)

DOCS_DIR = Path("docs")  # folder where .txt files stored
INDEX_FILE = Path("rag_index.joblib")
DOCS_FILE = Path("rag_docs.json")


class SimpleRAG:
    def __init__(self):
        self.vectorizer = None
        self.doc_texts = []
        self.doc_ids = []     # filenames only
        self.doc_titles = []

    def build_index_from_folder(self, folder: str = str(DOCS_DIR)):
        folder = Path(folder)
        files = sorted(folder.glob("*.txt"))

        docs = []
        ids = []
        titles = []

        for f in files:
            txt = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue

            docs.append(txt)
            ids.append(f.name)       # << FIXED (was relative path)
            titles.append(f.stem)

        if len(docs) == 0:
            self.vectorizer = None
            self.doc_texts = []
            self.doc_ids = []
            self.doc_titles = []
            return

        self.vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english"
        )

        X = self.vectorizer.fit_transform(docs)

        self.doc_texts = docs
        self.doc_ids = ids
        self.doc_titles = titles

        joblib.dump({
            "vectorizer": self.vectorizer,
            "doc_texts": self.doc_texts,
            "doc_ids": self.doc_ids,
            "doc_titles": self.doc_titles
        }, INDEX_FILE)

        DOCS_FILE.write_text(json.dumps({
            "doc_ids": self.doc_ids,
            "doc_titles": self.doc_titles,
            "n_docs": len(self.doc_ids)
        }, indent=2), encoding="utf-8")

    def load_index(self, docs_folder=None):
        if INDEX_FILE.exists():
            data = joblib.load(INDEX_FILE)
            self.vectorizer = data["vectorizer"]
            self.doc_texts = data["doc_texts"]
            self.doc_ids = data["doc_ids"]
            self.doc_titles = data["doc_titles"]
        else:
            # build automatically
            self.build_index_from_folder(str(DOCS_DIR))

    def retrieve(self, query: str, top_k: int = 5):
        if self.vectorizer is None:
            return []

        q_vec = self.vectorizer.transform([query])
        d_vec = self.vectorizer.transform(self.doc_texts)

        sims = cosine_similarity(q_vec, d_vec).flatten()
        idx = np.argsort(-sims)[:top_k]

        results = []
        for i in idx:
            score = float(sims[i])
            excerpt = self.doc_texts[i][:1200]  # limit excerpt
            results.append((self.doc_ids[i], excerpt, score))

        return results

    def get_context(self, query: str, top_k: int = 3):
        hits = self.retrieve(query, top_k=top_k)
        pieces = []
        for doc_id, excerpt, score in hits:
            pieces.append(
                f"Source: {doc_id}\nScore: {score:.3f}\n{excerpt}"
            )
        return "\n\n".join(pieces)


# shared instance
RAG = SimpleRAG()
