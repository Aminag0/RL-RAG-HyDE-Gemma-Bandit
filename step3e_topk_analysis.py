"""
Improvement 3 — Top-K Sensitivity Analysis
============================================
Paper improvement over arxiv 2506.21568:

The original paper fixes top_k=3 without justification.
We investigate how retrieval quality and latency change
across k = 1, 3, 5, 7 for both RAG and HyDE pipelines.

This answers: Is k=3 actually optimal?

Run:
    python step3e_topk_analysis.py
"""

import json
import os
import time
import requests
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
QDRANT_STORAGE = "qdrant_storage"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_FILE   = "data/results_topk.json"
PLOTS_DIR      = "data/plots"
TIMEOUT        = 120
MODEL          = "gemma3:1b"
K_VALUES       = [1, 3, 5, 7]

os.makedirs(PLOTS_DIR, exist_ok=True)

PHYSICS_QUESTIONS = [
    "What is the standard model of particle physics?",
    "Explain quantum entanglement and its implications.",
    "What is the Higgs boson and why is it important?",
    "How does general relativity describe gravity?",
    "What is dark matter and what evidence supports its existence?",
    "Explain the concept of quantum superposition.",
    "What is the Big Bang theory and what evidence supports it?",
    "How does nuclear fusion work in stars?",
    "What is the uncertainty principle in quantum mechanics?",
    "Explain the photoelectric effect and its significance.",
    "What is a black hole and how does it form?",
    "How does electromagnetic induction work?",
]

# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ── Retrieve chunks ───────────────────────────────────────────────────────────
def retrieve(query_vec, qdrant, top_k):
    results = qdrant.query_points(
        collection_name="physics_papers",
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points
    return results

# ── Generate hypothetical doc ─────────────────────────────────────────────────
def generate_hypothetical_doc(question):
    prompt = f"""Write a brief scientific passage answering: "{question}"
Write as if from a physics textbook. Keep under 80 words."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 120, "num_ctx": 512}},
            timeout=TIMEOUT
        )
        return r.json().get("response", "").strip()
    except:
        return question

# ── Call Ollama for latency ───────────────────────────────────────────────────
def call_ollama_timed(prompt, model=MODEL):
    start = time.time()
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 2048}},
            timeout=TIMEOUT
        )
        return time.time() - start
    except:
        return time.time() - start

# ── Main analysis ─────────────────────────────────────────────────────────────
def run_topk_analysis():
    print("="*60)
    print("Improvement 3 — Top-K Sensitivity Analysis")
    print("Paper: arxiv 2506.21568")
    print("="*60)

    print("\n[1/3] Loading models...")
    embedder = SentenceTransformer(EMBED_MODEL)
    qdrant   = QdrantClient(path=QDRANT_STORAGE)
    print(f"   Vectors: {qdrant.count('physics_papers').count}")

    print(f"\n[2/3] Testing k = {K_VALUES} for RAG and HyDE...")
    print(f"   Using {len(PHYSICS_QUESTIONS)} questions x {len(K_VALUES)} k values x 2 methods\n")

    # Pre-generate hypothetical docs (reuse across k values)
    print("   Pre-generating hypothetical documents...")
    hypo_docs = {}
    for q in PHYSICS_QUESTIONS:
        hypo_docs[q] = generate_hypothetical_doc(q)
        print(f"   Generated: {q[:50]}")

    results_by_k = {k: {"rag": [], "hyde": []} for k in K_VALUES}

    for k in K_VALUES:
        print(f"\n--- Testing k={k} ---")

        rag_relevances  = []
        hyde_relevances = []
        rag_latencies   = []
        hyde_latencies  = []

        for i, question in enumerate(PHYSICS_QUESTIONS):
            print(f"   [{i+1}/{len(PHYSICS_QUESTIONS)}] {question[:50]}", end="", flush=True)

            query_vec = embedder.encode(question)
            hypo_vec  = embedder.encode(hypo_docs[question])

            # RAG retrieval
            rag_results = retrieve(query_vec.tolist(), qdrant, k)
            rag_chunks  = [r.payload.get("text","") for r in rag_results]

            rag_rel = []
            for chunk in rag_chunks:
                chunk_vec = embedder.encode(chunk)
                rag_rel.append(cosine_similarity(query_vec, chunk_vec))
            rag_avg_rel = statistics.mean(rag_rel)
            rag_relevances.append(rag_avg_rel)

            # RAG latency — time full pipeline
            rag_context = "\n\n".join(rag_chunks)
            rag_prompt  = f"Use these passages:\n{rag_context}\n\nQuestion: {question}\nAnswer:"
            rag_lat     = call_ollama_timed(rag_prompt)
            rag_latencies.append(rag_lat)

            # HyDE retrieval
            hyde_results = retrieve(hypo_vec.tolist(), qdrant, k)
            hyde_chunks  = [r.payload.get("text","") for r in hyde_results]

            hyde_rel = []
            for chunk in hyde_chunks:
                chunk_vec = embedder.encode(chunk)
                hyde_rel.append(cosine_similarity(query_vec, chunk_vec))
            hyde_avg_rel = statistics.mean(hyde_rel)
            hyde_relevances.append(hyde_avg_rel)

            # HyDE latency
            hyde_context = "\n\n".join(hyde_chunks)
            hyde_prompt  = f"Use these passages:\n{hyde_context}\n\nQuestion: {question}\nAnswer:"
            hyde_lat     = call_ollama_timed(hyde_prompt)
            hyde_latencies.append(hyde_lat)

            print(f" | RAG rel={rag_avg_rel:.3f} lat={rag_lat:.1f}s | HyDE rel={hyde_avg_rel:.3f} lat={hyde_lat:.1f}s")

        results_by_k[k]["rag"] = {
            "avg_relevance": statistics.mean(rag_relevances),
            "avg_latency":   statistics.mean(rag_latencies),
            "relevances":    rag_relevances,
            "latencies":     rag_latencies,
        }
        results_by_k[k]["hyde"] = {
            "avg_relevance": statistics.mean(hyde_relevances),
            "avg_latency":   statistics.mean(hyde_latencies),
            "relevances":    hyde_relevances,
            "latencies":     hyde_latencies,
        }

    # Print summary table
    print("\n" + "="*60)
    print("TOP-K SENSITIVITY RESULTS")
    print("="*60)
    print(f"{'k':<6} {'Method':<8} {'Avg Relevance':<18} {'Avg Latency (s)':<18}")
    print("-"*50)
    for k in K_VALUES:
        for method in ["rag", "hyde"]:
            rel = results_by_k[k][method]["avg_relevance"]
            lat = results_by_k[k][method]["avg_latency"]
            print(f"{k:<6} {method.upper():<8} {rel:<18.4f} {lat:<18.2f}")

    # Find optimal k for each method
    print("\nOptimal k by relevance:")
    for method in ["rag", "hyde"]:
        best_k   = max(K_VALUES, key=lambda k: results_by_k[k][method]["avg_relevance"])
        best_rel = results_by_k[best_k][method]["avg_relevance"]
        print(f"   {method.upper()}: k={best_k} (relevance={best_rel:.4f})")

    print(f"\nKey finding: Does k=3 (paper's choice) achieve optimal relevance?")
    for method in ["rag", "hyde"]:
        best_k = max(K_VALUES, key=lambda k: results_by_k[k][method]["avg_relevance"])
        if best_k == 3:
            print(f"   {method.upper()}: YES — k=3 is optimal, validating paper's choice")
        else:
            print(f"   {method.upper()}: NO — k={best_k} is better, paper's choice suboptimal")

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_by_k, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {RESULTS_FILE}")

    return results_by_k


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_topk(results_by_k):
    print("\n[3/3] Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rag_rel  = [results_by_k[k]["rag"]["avg_relevance"]  for k in K_VALUES]
    hyde_rel = [results_by_k[k]["hyde"]["avg_relevance"] for k in K_VALUES]
    rag_lat  = [results_by_k[k]["rag"]["avg_latency"]    for k in K_VALUES]
    hyde_lat = [results_by_k[k]["hyde"]["avg_latency"]   for k in K_VALUES]

    # Plot 1 — Relevance vs K
    ax = axes[0]
    ax.plot(K_VALUES, rag_rel,  marker="o", color="#55A868", label="RAG",
            linewidth=2, markersize=8)
    ax.plot(K_VALUES, hyde_rel, marker="s", color="#C44E52", label="HyDE",
            linewidth=2, markersize=8)
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.7, label="Paper's k=3")

    for k, r, h in zip(K_VALUES, rag_rel, hyde_rel):
        ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#55A868")
        ax.annotate(f"{h:.3f}", (k, h), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="#C44E52")

    ax.set_xlabel("Number of Retrieved Chunks (k)", fontsize=11)
    ax.set_ylabel("Average Retrieval Relevance (Cosine Similarity)", fontsize=11)
    ax.set_title("Retrieval Relevance vs Top-K\n(higher is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 0.75)

    # Plot 2 — Latency vs K
    ax = axes[1]
    ax.plot(K_VALUES, rag_lat,  marker="o", color="#55A868", label="RAG",
            linewidth=2, markersize=8)
    ax.plot(K_VALUES, hyde_lat, marker="s", color="#C44E52", label="HyDE",
            linewidth=2, markersize=8)
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.7, label="Paper's k=3")

    for k, r, h in zip(K_VALUES, rag_lat, hyde_lat):
        ax.annotate(f"{r:.1f}s", (k, r), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#55A868")
        ax.annotate(f"{h:.1f}s", (k, h), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="#C44E52")

    ax.set_xlabel("Number of Retrieved Chunks (k)", fontsize=11)
    ax.set_ylabel("Average Response Latency (seconds)", fontsize=11)
    ax.set_title("Response Latency vs Top-K\n(lower is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_VALUES)

    plt.suptitle("Top-K Sensitivity Analysis: Relevance vs Latency Tradeoff\n(Our contribution — not in original paper)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "figure9_topk_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results_by_k = run_topk_analysis()
    plot_topk(results_by_k)

    print("\n" + "="*60)
    print("CONTRIBUTION SUMMARY")
    print("="*60)
    print("Original paper: Fixed top_k=3 with no sensitivity analysis")
    print("Our addition  : Tested k=1,3,5,7 measuring relevance + latency")
    print("Finding       : Shows optimal k and validates or challenges")
    print("               the paper's fixed choice of k=3")
    print("Plot saved    : data/plots/figure9_topk_sensitivity.png")


if __name__ == "__main__":
    main()
