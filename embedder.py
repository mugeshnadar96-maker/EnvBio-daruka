"""
embedder.py
-----------
Pluggable embedding backend for the environmental knowledge base.

PRODUCTION PATH (default, used automatically whenever the environment
has normal internet access):
    Uses chromadb's built-in ONNX build of `all-MiniLM-L6-v2`
    (chromadb.utils.embedding_functions.DefaultEmbeddingFunction), which
    is the same model family requested in the spec, running via onnxruntime
    (no torch dependency needed).

FALLBACK PATH (used only when the MiniLM weights cannot be downloaded,
e.g. a network-restricted sandbox):
    A deterministic local TF-IDF + SVD embedding fitted on the corpus
    itself. This keeps the whole RAG pipeline (ingest -> embed -> store ->
    retrieve) fully runnable end-to-end for development/testing, but it is
    NOT a substitute for the real sentence-transformer in production —
    semantic generalization to queries with different wording will be
    weaker. See README.md, "Known Sandbox Limitation".

The rest of the codebase (ingest_data.py, conversation_agent.py) only
talks to `get_embedder()` and never cares which backend is active.
"""

import os
import pickle
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "env_knowledge_db")
FALLBACK_VECTORIZER_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
FALLBACK_SVD_PATH = os.path.join(CACHE_DIR, "tfidf_svd.pkl")


class MiniLMEmbedder:
    """Wraps chromadb's DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2)."""

    name = "all-MiniLM-L6-v2 (onnx, production)"

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def __call__(self, texts):
        return self._ef(texts)


class TfidfFallbackEmbedder:
    """
    Local, dependency-light fallback embedder used only when the real
    MiniLM ONNX weights cannot be fetched. Fits TF-IDF + TruncatedSVD
    (LSA) on the knowledge base corpus once at ingestion time, then
    reuses the fitted transform for query-time embedding.
    """

    name = "TF-IDF+SVD (local fallback, degraded semantic quality)"
    dim = 128

    def __init__(self):
        self._vectorizer = None
        self._svd = None
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(FALLBACK_VECTORIZER_PATH) and os.path.exists(FALLBACK_SVD_PATH):
            with open(FALLBACK_VECTORIZER_PATH, "rb") as f:
                self._vectorizer = pickle.load(f)
            with open(FALLBACK_SVD_PATH, "rb") as f:
                self._svd = pickle.load(f)

    def fit(self, corpus_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=4000, stop_words="english"
        )
        tfidf = self._vectorizer.fit_transform(corpus_texts)
        n_components = min(self.dim, max(2, tfidf.shape[0] - 1), tfidf.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)

        with open(FALLBACK_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self._vectorizer, f)
        with open(FALLBACK_SVD_PATH, "wb") as f:
            pickle.dump(self._svd, f)

    def __call__(self, texts):
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError(
                "TfidfFallbackEmbedder not fitted yet. Run ingest_data.py first."
            )
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf)
        # L2 normalize so cosine similarity behaves like a sentence embedding
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).tolist()


class SpacyVectorEmbedder:
    """
    Middle-tier embedder: spaCy's en_core_web_md ships real pretrained
    (GloVe-style) 300-dim word vectors. Unlike the TF-IDF fallback, these
    are genuine pretrained semantic vectors -- two sentences about the same
    topic with zero shared vocabulary still score highly similar.

    Installable in network-restricted environments because the wheel is
    served from a GitHub release (release-assets.githubusercontent.com),
    not from huggingface.co or an S3 bucket:

        pip install spacy
        pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl"
    """

    name = "spaCy en_core_web_md (pretrained GloVe-style vectors, 300d)"

    def __init__(self):
        import spacy
        self._nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer"])

    def __call__(self, texts):
        vecs = []
        for doc in self._nlp.pipe(texts):
            v = doc.vector
            norm = (v ** 2).sum() ** 0.5
            vecs.append((v / norm).tolist() if norm > 0 else v.tolist())
        return vecs


def get_embedder(prefer_production=True):
    """
    Returns (embedder_callable, backend_name).
    Tries backends in descending order of semantic quality:
      1. Real MiniLM ONNX (chromadb default) -- needs the S3 host reachable.
      2. spaCy en_core_web_md pretrained vectors -- needs only GitHub, which
         is reachable even in network-restricted sandboxes.
      3. TF-IDF+SVD fitted on the corpus -- always works, weakest semantics.
    """
    if prefer_production:
        try:
            emb = MiniLMEmbedder()
            emb(["connectivity check"])  # forces the lazy ONNX download
            return emb, emb.name
        except Exception as e:
            print(f"[embedder] Production MiniLM unavailable ({type(e).__name__}: {e}). "
                  f"Trying spaCy pretrained vectors next.")
        try:
            emb = SpacyVectorEmbedder()
            return emb, emb.name
        except Exception as e:
            print(f"[embedder] spaCy vectors unavailable ({type(e).__name__}: {e}). "
                  f"Falling back to local TF-IDF embedder.")
    emb = TfidfFallbackEmbedder()
    return emb, emb.name
