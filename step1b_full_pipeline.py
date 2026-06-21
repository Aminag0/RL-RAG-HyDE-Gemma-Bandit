"""
Cleanup + Full PDF Download + Re-ingest
=========================================
1. Deletes existing Qdrant collections
2. Deletes abstract txt files
3. Downloads full PDFs from arXiv
4. Extracts full text using pymupdf
5. Re-ingests into Qdrant with full content

Install:
    pip install pymupdf requests qdrant-client sentence-transformers pymongo

Run:
    python step1b_full_pipeline.py
"""

import os
import json
import time
import uuid
import shutil
import requests
from pathlib import Path

import fitz  # pymupdf
import pymongo
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
PAPERS_DIR      = "data/physics_papers"
PDF_DIR         = "data/physics_pdfs"
FULL_TEXT_DIR   = "data/physics_fulltext"
META_FILE       = "data/papers_metadata.json"
QDRANT_STORAGE  = "qdrant_storage"
COLLECTION_NAME = "physics_papers"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
MONGO_URI       = "mongodb://localhost:27017"
MONGO_DB        = "jarvis"
CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 50
DELAY           = 2.0   # seconds between PDF downloads

# ── Step 1: Cleanup ───────────────────────────────────────────────────────────
def cleanup():
    print("="*60)
    print("Step 1 - Cleanup")
    print("="*60)

    # Delete abstract txt files
    if os.path.exists(PAPERS_DIR):
        count = len(list(Path(PAPERS_DIR).glob("*.txt")))
        shutil.rmtree(PAPERS_DIR)
        print(f"   Deleted {count} abstract txt files from {PAPERS_DIR}/")
    else:
        print(f"   {PAPERS_DIR}/ not found, skipping")

    # Delete Qdrant storage
    if os.path.exists(QDRANT_STORAGE):
        shutil.rmtree(QDRANT_STORAGE)
        print(f"   Deleted Qdrant storage: {QDRANT_STORAGE}/")
    else:
        print(f"   {QDRANT_STORAGE}/ not found, skipping")

    # Recreate directories
    os.makedirs(PAPERS_DIR,    exist_ok=True)
    os.makedirs(PDF_DIR,       exist_ok=True)
    os.makedirs(FULL_TEXT_DIR, exist_ok=True)
    print("   Recreated clean directories.")


# ── Step 2: Download PDFs ─────────────────────────────────────────────────────
def download_pdfs():
    print("\n" + "="*60)
    print("Step 2 - Downloading Full PDFs from arXiv")
    print("="*60)

    # Load metadata
    with open(META_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"   Total papers to download: {len(papers)}")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; mailto:student@example.com)"
    }

    downloaded = 0
    failed     = []

    for i, paper in enumerate(papers):
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            continue

        safe_id  = arxiv_id.replace("/", "_")
        pdf_path = os.path.join(PDF_DIR, f"{safe_id}.pdf")

        # Skip if already downloaded
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            downloaded += 1
            continue

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        try:
            r = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
            r.raise_for_status()

            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = os.path.getsize(pdf_path) / 1024
            downloaded += 1
            print(f"   [{i+1}/{len(papers)}] {arxiv_id} — {size_kb:.0f}KB")

        except Exception as e:
            print(f"   [{i+1}/{len(papers)}] FAILED {arxiv_id}: {e}")
            failed.append(arxiv_id)

        time.sleep(DELAY)

    print(f"\n   Downloaded : {downloaded}/{len(papers)}")
    print(f"   Failed     : {len(failed)}")
    if failed:
        print(f"   Failed IDs : {failed[:5]}...")

    return downloaded


# ── Step 3: Extract full text from PDFs ───────────────────────────────────────
def extract_text():
    print("\n" + "="*60)
    print("Step 3 - Extracting Full Text from PDFs")
    print("="*60)

    with open(META_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)

    papers_by_id = {p["arxiv_id"]: p for p in papers}
    pdf_files    = list(Path(PDF_DIR).glob("*.pdf"))
    extracted    = 0

    print(f"   PDF files found: {len(pdf_files)}")

    for i, pdf_path in enumerate(pdf_files):
        safe_id  = pdf_path.stem
        arxiv_id = safe_id.replace("_", "/", 1) if "_" in safe_id else safe_id
        txt_path = os.path.join(FULL_TEXT_DIR, f"{safe_id}.txt")

        # Skip if already extracted
        if os.path.exists(txt_path):
            extracted += 1
            continue

        paper = papers_by_id.get(arxiv_id, {})

        try:
            doc  = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Clean text
            import re
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = text.strip()

            if len(text) < 100:
                raise ValueError("Extracted text too short — likely scanned PDF")

            # Save with metadata header
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"TITLE: {paper.get('title','')}\n")
                f.write(f"AUTHORS: {', '.join(paper.get('authors', []))}\n")
                f.write(f"DATE: {paper.get('published','')}\n")
                f.write(f"ARXIV_ID: {arxiv_id}\n")
                f.write(f"PDF: https://arxiv.org/pdf/{arxiv_id}\n")
                f.write("\n" + "="*60 + "\n\n")
                f.write(text[:50000])  # cap at 50k chars per paper

            extracted += 1
            if (i + 1) % 50 == 0:
                print(f"   Extracted {i+1}/{len(pdf_files)} papers...")

        except Exception as e:
            # Fall back to abstract
            abstract = paper.get("summary", paper.get("abstract", ""))
            if abstract:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"TITLE: {paper.get('title','')}\n")
                    f.write(f"ARXIV_ID: {arxiv_id}\n")
                    f.write("\n" + "="*60 + "\n\n")
                    f.write(abstract)
                extracted += 1

    print(f"   Extracted: {extracted}/{len(pdf_files)} papers")
    return extracted


# ── Step 4: Chunk text ────────────────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks if chunks else [text]


def parse_full_text(filepath):
    import re
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    parts = content.split("=" * 60)
    body  = parts[1].strip() if len(parts) > 1 else content

    return {
        "arxiv_id": field("ARXIV_ID"),
        "title":    field("TITLE"),
        "authors":  field("AUTHORS"),
        "date":     field("DATE"),
        "pdf_url":  field("PDF"),
        "body":     body,
        "filename": os.path.basename(filepath),
    }


# ── Step 5: Re-ingest into Qdrant ─────────────────────────────────────────────
def reingest_qdrant(embedder, client):
    print("\n" + "="*60)
    print("Step 5 - Re-ingesting Full Text into Qdrant")
    print("="*60)

    dim = embedder.get_embedding_dimension()

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    txt_files    = sorted(Path(FULL_TEXT_DIR).glob("*.txt"))
    all_points   = []
    total_chunks = 0

    print(f"   Processing {len(txt_files)} full-text files...")

    for i, fp in enumerate(txt_files):
        try:
            paper  = parse_full_text(str(fp))
            chunks = chunk_text(paper["body"])
            embeds = embedder.encode(chunks, show_progress_bar=False)
            total_chunks += len(chunks)

            for j, (chunk, emb) in enumerate(zip(chunks, embeds)):
                all_points.append(PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = emb.tolist(),
                    payload = {
                        "arxiv_id":    paper["arxiv_id"],
                        "title":       paper["title"],
                        "authors":     paper["authors"],
                        "date":        paper["date"],
                        "pdf_url":     paper["pdf_url"],
                        "chunk_index": j,
                        "text":        chunk,
                        "filename":    paper["filename"],
                    }
                ))

            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(txt_files)} papers...")

        except Exception as e:
            print(f"   Error {fp.name}: {e}")

    # Upload in batches
    BATCH = 200
    for s in range(0, len(all_points), BATCH):
        client.upsert(COLLECTION_NAME, all_points[s:s+BATCH])
        print(f"   Uploaded {min(s+BATCH, len(all_points))}/{len(all_points)} vectors...")

    print(f"\n   Total papers   : {len(txt_files)}")
    print(f"   Total chunks   : {total_chunks}")
    print(f"   Vectors stored : {len(all_points)}")
    return len(all_points)


# ── Step 6: Re-ingest personal data ──────────────────────────────────────────
def reingest_personal(embedder, client):
    print("\n" + "="*60)
    print("Step 6 - Re-ingesting Personal Data")
    print("="*60)

    PERSONAL_COLLECTION = "personal_data"
    dim = embedder.get_embedding_dimension()

    client.create_collection(
        collection_name=PERSONAL_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    txt_path = "data/personal/personal_data.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text(content, size=50, overlap=10)
    embeds = embedder.encode(chunks, show_progress_bar=False)

    points = []
    for j, (chunk, emb) in enumerate(zip(chunks, embeds)):
        points.append(PointStruct(
            id      = str(uuid.uuid4()),
            vector  = emb.tolist(),
            payload = {"text": chunk, "source": "personal_data", "chunk_index": j}
        ))

    client.upsert(PERSONAL_COLLECTION, points)
    print(f"   Personal data vectors: {len(points)}")


# ── Verification ──────────────────────────────────────────────────────────────
def verify(client):
    print("\n" + "="*60)
    print("Verification")
    print("="*60)

    physics_count  = client.count("physics_papers").count
    personal_count = client.count("personal_data").count
    print(f"   physics_papers : {physics_count} vectors")
    print(f"   personal_data  : {personal_count} vectors")

    # Test query
    embedder = SentenceTransformer(EMBED_MODEL)
    qvec     = embedder.encode("quantum entanglement").tolist()
    results  = client.query_points(
        collection_name="physics_papers",
        query=qvec, limit=3, with_payload=True
    ).points

    print(f"\n   Test query: 'quantum entanglement'")
    for r in results:
        print(f"   [{r.score:.3f}] {r.payload.get('title','')[:60]}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Full Pipeline: Cleanup + PDF Download + Re-ingest")
    print("Paper: arxiv 2506.21568")
    print("="*60)

    # Cleanup
    cleanup()

    # Download PDFs
    download_pdfs()

    # Extract text
    extract_text()

    # Load models
    print("\n[MODELS] Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"   Dim: {embedder.get_embedding_dimension()}")

    print("[MODELS] Connecting to Qdrant...")
    client = QdrantClient(path=QDRANT_STORAGE)

    # Re-ingest
    reingest_qdrant(embedder, client)
    reingest_personal(embedder, client)

    # Verify
    verify(client)

    print("\n" + "="*60)
    print("Done. Ready for Step 3 - RAG vs HyDE pipelines.")
    print("="*60)


if __name__ == "__main__":
    main()
