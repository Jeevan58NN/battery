# rag.py
import os
import glob
import json
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import nltk
from tqdm import tqdm

# ensure nltk tokenizers downloaded (only first run)
nltk_packages = ["punkt", "stopwords"]
for p in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{p}")
    except Exception:
        nltk.download(p)

DOCS_DIR = Path("docs")           # put your .txt files here
INDEX_FILE = Path("rag_index.joblib")
DOCS_FILE  = Path("rag_docs.json")

class SimpleRAG:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_texts: List[str] = []
        self.doc_ids: List[str] = []  # filenames or IDs
        self.doc_titles: List[str] = []

    def build_index_from_folder(self, folder: str = str(DOCS_DIR)):
        folder = Path(folder)
        files = sorted(folder.glob("**/*.txt"))
        docs = []
        ids = []
        titles = []
        for f in files:
            txt = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            docs.append(txt)
            ids.append(str(f.relative_to(Path.cwd())))
            titles.append(f.name)
        if len(docs) == 0:
            # no docs found
            self.vectorizer = None
            self.doc_texts = []
            self.doc_ids = []
            self.doc_titles = []
            return

        # Build TF-IDF vectorizer with sensible limits
        self.vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english"
        )
        X = self.vectorizer.fit_transform(docs)
        self.doc_texts = docs
        self.doc_ids = ids
        self.doc_titles = titles

        # Save index
        joblib.dump({
            "vectorizer": self.vectorizer,
            "doc_texts": self.doc_texts,
            "doc_ids": self.doc_ids,
            "doc_titles": self.doc_titles
        }, INDEX_FILE)

        # Also save raw docs metadata for debug
        DOCS_FILE.write_text(json.dumps({
            "doc_ids": self.doc_ids,
            "doc_titles": self.doc_titles,
            "n_docs": len(self.doc_ids)
        }, indent=2), encoding="utf-8")

    def load_index(self):
        if INDEX_FILE.exists():
            data = joblib.load(INDEX_FILE)
            self.vectorizer = data.get("vectorizer")
            self.doc_texts = data.get("doc_texts", [])
            self.doc_ids = data.get("doc_ids", [])
            self.doc_titles = data.get("doc_titles", [])
        else:
            # try building automatically if folder exists
            if DOCS_DIR.exists():
                self.build_index_from_folder(str(DOCS_DIR))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Return list of (doc_id, doc_text_excerpt, score)
        """
        if self.vectorizer is None:
            return []

        qv = self.vectorizer.transform([query])
        D = self.vectorizer.transform(self.doc_texts)
        sims = cosine_similarity(qv, D).flatten()
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            score = float(sims[i])
            text = self.doc_texts[i]
            doc_id = self.doc_ids[i]
            excerpt = (text[:1000] + "...") if len(text) > 1000 else text
            results.append((doc_id, excerpt, score))
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        hits = self.retrieve(query, top_k=top_k)
        pieces = []
        for doc_id, excerpt, score in hits:
            pieces.append(f"Source: {doc_id}\nScore: {score:.3f}\n{excerpt}")
        return "\n\n".join(pieces)


# single shared instance (import from app)
RAG = SimpleRAG()
