"""
ingest_data.py
--------------
Loads the environmental knowledge base (soil health, land use, biodiversity,
climate, human-impact findings drawn from SoilHealthDB-V2, OpenLandMap-soildb,
LUCAS, FAO "Recarbonizing Global Soils", OECD Farmland Habitat Biodiversity
Indicator guidelines, and IPCC AR6 WGIII Ch.7), converts each record into a
retrievable document, chunks it, embeds it, and persists it to a local
ChromaDB collection.

Usage:
    python3 ingest_data.py [--csv data/knowledge_base.csv] [--reset]

Document template (per spec):
    "Context: {situation}. Finding: {result}. Metric: {affected_metric}.
     Impact: {quantitative_change}. Evidence: {source}."
"""

import argparse
import os
import textwrap

import pandas as pd
import chromadb

from embedder import get_embedder

DB_DIR = os.path.join(os.path.dirname(__file__), "env_knowledge_db")
COLLECTION_NAME = "environmental_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def row_to_document(row) -> str:
    return (
        f"Context: {row['context']}. "
        f"Finding: {row['finding']}. "
        f"Metric: {row['metric']}. "
        f"Impact: {row['quantitative_impact']}. "
        f"Evidence: {row['source']} {row['year']} [{row['source_id']}]."
    )


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple sliding-window character chunker (most knowledge-base
    documents here are well under 500 chars, so this is normally a no-op;
    it matters once full dataset rows / long-form PDF excerpts are ingested)."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_documents(df: pd.DataFrame):
    docs, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        full_doc = row_to_document(row)
        for i, chunk in enumerate(chunk_text(full_doc)):
            docs.append(chunk)
            metadatas.append({
                "domain": row["domain"],
                "metric": row["metric"],
                "source": row["source"],
                "source_id": str(row["source_id"]),
                "year": str(row["year"]),
                "record_id": str(row["id"]),
            })
            ids.append(f"rec{row['id']}_chunk{i}")
    return docs, metadatas, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv"))
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} source records from {args.csv}")
    print(f"Domains: {df['domain'].value_counts().to_dict()}")

    docs, metadatas, ids = build_documents(df)
    print(f"Built {len(docs)} chunked documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embed_fn, backend_name = get_embedder()
    print(f"Embedding backend: {backend_name}")

    # If we're on the fallback embedder, it must be *fit* on the corpus first.
    if hasattr(embed_fn, "fit") and not hasattr(embed_fn, "_ef"):
        embed_fn.fit(docs)

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_backend": backend_name},
    )

    embeddings = embed_fn(docs)

    # Upsert in batches
    BATCH = 64
    for i in range(0, len(docs), BATCH):
        collection.upsert(
            documents=docs[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
        )

    print(f"Stored {collection.count()} documents in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {DB_DIR}")

    print("\nSample stored document:")
    print(textwrap.fill(docs[0], 100))
    print(metadatas[0])


if __name__ == "__main__":
    main()
