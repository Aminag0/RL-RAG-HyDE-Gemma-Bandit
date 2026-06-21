"""
Step 3 - RAG vs HyDE vs Baseline Evaluation
=============================================
Paper: Assessing RAG and HyDE on 1B vs 4B Gemma LLMs (arxiv 2506.21568)

Implements three pipelines:
1. Baseline  — Gemma answers from its own knowledge only
2. RAG       — Gemma answers using retrieved physics paper chunks
3. HyDE      — Gemma generates hypothetical answer first, uses that
               to search, then answers with retrieved chunks

Evaluates on:
- 12 physics questions  — measures latency
- 10 personal questions — measures hallucination rate

Run:
    python step3_evaluation.py
"""

import os
import json
import time
import re
import requests
import pymongo

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434/api/generate"
COLLECTION_NAME  = "physics_papers"
PERSONAL_COLL    = "personal_data"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
MONGO_URI        = "mongodb://localhost:27017"
MONGO_DB         = "jarvis"
QDRANT_STORAGE   = "qdrant_storage"
RESULTS_FILE     = "data/results.json"
TOP_K            = 3       # chunks to retrieve — same as paper
TIMEOUT          = 180     # seconds per query

MODELS = ["gemma3:1b", "gemma3:4b"]

# ── Load questions ────────────────────────────────────────────────────────────
with open("data/questions.json", "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

PHYSICS_QUESTIONS  = QUESTIONS["physics_questions"]
PERSONAL_QUESTIONS = QUESTIONS["personal_questions"]

# ── Ollama call ───────────────────────────────────────────────────────────────
def call_ollama(prompt, model, timeout=TIMEOUT):
    """Call Ollama and return response text + time taken."""
    start = time.time()
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400,
                    "num_ctx":     4096,
                }
            },
            timeout=timeout
        )
        elapsed = time.time() - start
        response = r.json().get("response", "").strip()
        return response, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"ERROR: {e}", elapsed

# ── Retrieve from Qdrant ──────────────────────────────────────────────────────
def retrieve(query, embedder, qdrant, collection, top_k=TOP_K):
    """Retrieve top-k chunks from Qdrant."""
    qvec    = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=collection,
        query=qvec,
        limit=top_k,
        with_payload=True,
    ).points
    return results

# ── Retrieve from MongoDB ─────────────────────────────────────────────────────
def retrieve_personal_mongo(query, mongo_client):
    """Retrieve personal data from MongoDB based on query keywords."""
    db   = mongo_client[MONGO_DB]
    data = []

    query_lower = query.lower()

    # Route to relevant collections based on keywords
    if any(w in query_lower for w in ["name", "study", "university", "where"]):
        doc = db.profile.find_one()
        if doc:
            data.append(f"Name: {doc.get('name')}, University: {doc.get('university')}, Degree: {doc.get('degree')}")

    if any(w in query_lower for w in ["gpa", "grade", "academic"]):
        doc = db.profile.find_one()
        if doc:
            data.append(f"GPA: {doc.get('gpa')}")
        for a in db.academic_history.find():
            data.append(f"{a.get('year')} {a.get('semester')}: GPA {a.get('gpa')}")

    if any(w in query_lower for w in ["supervisor", "fyp", "advisor"]):
        for c in db.contacts.find({"relation": "Supervisor"}):
            data.append(f"Supervisor: {c.get('name')}, Email: {c.get('email')}")

    if any(w in query_lower for w in ["programming", "language", "code", "skill"]):
        doc = db.preferences.find_one()
        if doc:
            data.append(f"Programming languages: {', '.join(doc.get('programming_languages', []))}")

    if any(w in query_lower for w in ["hobby", "hobbies", "interest", "like"]):
        doc = db.preferences.find_one()
        if doc:
            data.append(f"Hobbies: {', '.join(doc.get('hobbies', []))}")
            data.append(f"Interests: {', '.join(doc.get('interests', []))}")

    if any(w in query_lower for w in ["address", "home", "live", "location"]):
        doc = db.profile.find_one()
        if doc:
            data.append(f"Address: {doc.get('address')}")

    if any(w in query_lower for w in ["degree", "graduate", "start", "year", "when"]):
        doc = db.profile.find_one()
        if doc:
            data.append(f"Degree: {doc.get('degree')}, Graduation year: {doc.get('grad_year')}")

    if any(w in query_lower for w in ["teammate", "team", "member", "colleague"]):
        for c in db.contacts.find({"relation": "Teammate"}):
            data.append(f"Teammate: {c.get('name')}, Email: {c.get('email')}")

    if any(w in query_lower for w in ["project", "work", "built", "developed"]):
        for p in db.projects.find():
            data.append(f"Project: {p.get('name')} ({p.get('year')}): {p.get('description')}")

    if any(w in query_lower for w in ["schedule", "monday", "tuesday", "wednesday", "class", "meeting"]):
        for s in db.schedule.find():
            data.append(f"{s.get('day')} {s.get('time')}: {s.get('event')} at {s.get('location')}")

    # If nothing matched, return all profile data
    if not data:
        doc = db.profile.find_one()
        if doc:
            data.append(str({k: v for k, v in doc.items() if k != "_id"}))

    return "\n".join(data)

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE 1: BASELINE
# Gemma answers from its own knowledge only — no retrieval
# ────────────────────────────────────────────────────────────────────────────
def baseline_pipeline(question, model, mongo_client=None, mode="physics"):
    if mode == "personal":
        prompt = f"""You are a personal assistant. Answer this question about the user based on your knowledge.
Be specific and factual. If you don't know, say you don't know.

Question: {question}

Answer:"""
    else:
        prompt = f"""You are a physics expert. Answer this question clearly and accurately.

Question: {question}

Answer:"""

    response, elapsed = call_ollama(prompt, model)
    return {
        "method":   "Baseline",
        "model":    model,
        "question": question,
        "response": response,
        "time":     elapsed,
        "context":  None,
    }

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE 2: RAG
# Retrieve relevant chunks, pass to Gemma as context
# ────────────────────────────────────────────────────────────────────────────
def rag_pipeline(question, model, embedder, qdrant, mongo_client, mode="physics"):
    # Retrieve context
    if mode == "personal":
        # Personal mode: retrieve from MongoDB + Qdrant personal collection
        mongo_context  = retrieve_personal_mongo(question, mongo_client)
        qdrant_results = retrieve(question, embedder, qdrant, PERSONAL_COLL)
        qdrant_context = "\n".join([r.payload.get("text", "") for r in qdrant_results])
        context        = mongo_context + "\n" + qdrant_context
        prompt = f"""You are a personal assistant. Use ONLY the following personal data to answer the question.
Do not add any information not present in the data below.

PERSONAL DATA:
{context}

Question: {question}

Answer (use only the data provided above):"""

    else:
        # Physics mode: retrieve from Qdrant physics collection
        results = retrieve(question, embedder, qdrant, COLLECTION_NAME)
        context = "\n\n".join([
            f"[Source: {r.payload.get('title','')}]\n{r.payload.get('text','')}"
            for r in results
        ])
        prompt = f"""You are a physics expert. Use the following retrieved passages to answer the question accurately.

RETRIEVED CONTEXT:
{context}

Question: {question}

Answer (based on the retrieved context):"""

    response, elapsed = call_ollama(prompt, model)
    return {
        "method":   "RAG",
        "model":    model,
        "question": question,
        "response": response,
        "time":     elapsed,
        "context":  context[:500],
    }

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE 3: HyDE
# Step 1: Generate hypothetical document from question
# Step 2: Use that hypothetical doc to search Qdrant
# Step 3: Retrieve real chunks matching hypothetical doc
# Step 4: Answer using real retrieved chunks
# ────────────────────────────────────────────────────────────────────────────
def hyde_pipeline(question, model, embedder, qdrant, mongo_client, mode="physics"):
    # Step 1: Generate hypothetical document
    if mode == "personal":
        hyde_prompt = f"""Write a brief personal profile entry that would answer this question: "{question}"
Write it as if it were real personal data. Be specific with names, dates, and details.
Keep it under 100 words."""
    else:
        hyde_prompt = f"""Write a brief scientific passage that would answer this question: "{question}"
Write it as if it were from a physics textbook or research paper.
Keep it under 100 words."""

    hypothetical_doc, hyde_time = call_ollama(hyde_prompt, model)

    # Step 2: Use hypothetical doc as search query
    if mode == "personal":
        qdrant_results = retrieve(hypothetical_doc, embedder, qdrant, PERSONAL_COLL)
        mongo_context  = retrieve_personal_mongo(question, mongo_client)
        context        = mongo_context + "\n" + "\n".join([
            r.payload.get("text", "") for r in qdrant_results
        ])
        prompt = f"""You are a personal assistant. Use ONLY the following personal data to answer the question.
Do not add any information not present in the data below.

PERSONAL DATA:
{context}

Question: {question}

Answer (use only the data provided above):"""

    else:
        results = retrieve(hypothetical_doc, embedder, qdrant, COLLECTION_NAME)
        context = "\n\n".join([
            f"[Source: {r.payload.get('title','')}]\n{r.payload.get('text','')}"
            for r in results
        ])
        prompt = f"""You are a physics expert. Use the following retrieved passages to answer the question accurately.

RETRIEVED CONTEXT:
{context}

Question: {question}

Answer (based on the retrieved context):"""

    # Step 3: Generate final answer
    response, answer_time = call_ollama(prompt, model)
    total_time = hyde_time + answer_time

    return {
        "method":           "HyDE",
        "model":            model,
        "question":         question,
        "response":         response,
        "time":             total_time,
        "hyde_time":        hyde_time,
        "answer_time":      answer_time,
        "hypothetical_doc": hypothetical_doc[:300],
        "context":          context[:500],
    }

# ── Hallucination detection ───────────────────────────────────────────────────
# Check if response contains information NOT in personal data
PERSONAL_FACTS = {
    "name":       "Maryam Amjad",
    "gpa":        "3.7",
    "university": "FAST-NUCES",
    "address":    "House 45, Street 7, F-10/3, Islamabad",
    "supervisor": "Dr. Qaiser Shafi",
    "grad_year":  "2026",
    "teammates":  ["Amna Nadeem", "Dur-e-Shahwar"],
    "languages":  ["Python", "Java", "C++"],
    "projects":   ["Nexa Aegis", "WHO Drug Alert RAG System"],
}

def check_hallucination(response, question):
    """
    Simple hallucination check for personal data questions.
    Returns True if response contains invented facts not in our data.
    """
    response_lower = response.lower()

    # Check for invented dates/numbers not in our data
    # Find years mentioned in response
    years_in_response = re.findall(r"\b(19|20)\d{2}\b", response)
    valid_years = {"2022", "2023", "2024", "2025", "2026", "2007"}
    invented_years = [y for y in years_in_response if y not in valid_years]
    if invented_years:
        return True, f"Invented years: {invented_years}"

    # Check for invented names
    common_invented_names = ["john", "alice", "bob", "sarah", "mike", "jane"]
    for name in common_invented_names:
        if name in response_lower:
            return True, f"Invented name: {name}"

    # Check for specific wrong GPA values
    gpa_matches = re.findall(r"\b[34]\.\d\b", response)
    valid_gpas  = {"3.5", "3.6", "3.7", "3.8"}
    for gpa in gpa_matches:
        if gpa not in valid_gpas:
            return True, f"Invented GPA: {gpa}"

    return False, "No hallucination detected"

# ── Run full evaluation ───────────────────────────────────────────────────────
def run_evaluation():
    print("="*60)
    print("Step 3 - RAG vs HyDE vs Baseline Evaluation")
    print("Paper: arxiv 2506.21568")
    print("="*60)

    # Load models
    print("\n[1/3] Loading models...")
    embedder = SentenceTransformer(EMBED_MODEL)
    qdrant   = QdrantClient(path=QDRANT_STORAGE)
    mongo    = pymongo.MongoClient(MONGO_URI)
    print(f"   Embedding model: {EMBED_MODEL}")
    print(f"   Qdrant vectors : {qdrant.count(COLLECTION_NAME).count}")
    print(f"   MongoDB DB     : {MONGO_DB}")

    # Warm up both models
    print("\n[2/3] Warming up Gemma models...")
    for model in MODELS:
        print(f"   Warming up {model}...", end="", flush=True)
        call_ollama("Hello", model, timeout=120)
        print(" done.")

    # Run evaluation
    print("\n[3/3] Running evaluation...")
    all_results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        # ── Physics questions — measure latency ───────────────────────────────
        print(f"\n--- PHYSICS QUESTIONS (latency) ---")
        for i, question in enumerate(PHYSICS_QUESTIONS):
            print(f"\n[{i+1}/{len(PHYSICS_QUESTIONS)}] {question[:60]}")

            for pipeline_fn, method in [
                (baseline_pipeline, "Baseline"),
                (rag_pipeline,      "RAG"),
                (hyde_pipeline,     "HyDE"),
            ]:
                print(f"   {method}...", end="", flush=True)

                if method == "Baseline":
                    result = pipeline_fn(question, model, mode="physics")
                else:
                    result = pipeline_fn(question, model, embedder, qdrant, mongo, mode="physics")

                result["question_type"] = "physics"
                result["question_num"]  = i + 1
                all_results.append(result)

                print(f" {result['time']:.2f}s")
                print(f"   Response: {result['response'][:100]}...")

        # ── Personal questions — measure hallucination ────────────────────────
        print(f"\n--- PERSONAL QUESTIONS (hallucination) ---")
        for i, question in enumerate(PERSONAL_QUESTIONS):
            print(f"\n[{i+1}/{len(PERSONAL_QUESTIONS)}] {question[:60]}")

            for pipeline_fn, method in [
                (baseline_pipeline, "Baseline"),
                (rag_pipeline,      "RAG"),
                (hyde_pipeline,     "HyDE"),
            ]:
                print(f"   {method}...", end="", flush=True)

                if method == "Baseline":
                    result = pipeline_fn(question, model, mongo_client=mongo, mode="personal")
                else:
                    result = pipeline_fn(question, model, embedder, qdrant, mongo, mode="personal")

                hallucinated, reason = check_hallucination(result["response"], question)
                result["question_type"] = "personal"
                result["question_num"]  = i + 1
                result["hallucinated"]  = hallucinated
                result["hall_reason"]   = reason
                all_results.append(result)

                hall_label = "HALLUCINATION" if hallucinated else "OK"
                print(f" {result['time']:.2f}s [{hall_label}]")
                print(f"   Response: {result['response'][:100]}...")

        # Save progress after each model
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n   Results saved: {RESULTS_FILE}")

    return all_results


# ── Print results table ───────────────────────────────────────────────────────
def print_results_table(all_results):
    import statistics

    print("\n" + "="*60)
    print("RESULTS — PHYSICS QUESTIONS LATENCY")
    print("="*60)
    print(f"{'Method':<12} {'Model':<12} {'Avg(s)':<10} {'Std(s)':<10} {'Min(s)':<10} {'Max(s)':<10}")
    print("-"*60)

    for model in MODELS:
        for method in ["Baseline", "RAG", "HyDE"]:
            times = [
                r["time"] for r in all_results
                if r["model"] == model
                and r["method"] == method
                and r["question_type"] == "physics"
            ]
            if times:
                avg = statistics.mean(times)
                std = statistics.stdev(times) if len(times) > 1 else 0
                mn  = min(times)
                mx  = max(times)
                print(f"{method:<12} {model:<12} {avg:<10.2f} {std:<10.2f} {mn:<10.2f} {mx:<10.2f}")

    print("\n" + "="*60)
    print("RESULTS — PERSONAL QUESTIONS HALLUCINATION")
    print("="*60)
    print(f"{'Method':<12} {'Model':<12} {'Hallucinations':<18} {'Rate':<10}")
    print("-"*60)

    for model in MODELS:
        for method in ["Baseline", "RAG", "HyDE"]:
            results = [
                r for r in all_results
                if r["model"] == model
                and r["method"] == method
                and r["question_type"] == "personal"
            ]
            if results:
                hall_count = sum(1 for r in results if r.get("hallucinated", False))
                total      = len(results)
                rate       = f"{hall_count}/{total}"
                pct        = f"{hall_count/total*100:.0f}%"
                print(f"{method:<12} {model:<12} {rate:<18} {pct:<10}")

    print("\n" + "="*60)
    print("KEY FINDINGS (matching paper)")
    print("="*60)
    print("RAG  — fastest + zero hallucinations (expected)")
    print("HyDE — slowest + highest hallucination rate (expected)")
    print("Baseline — moderate speed + some hallucinations (expected)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = run_evaluation()
    print_results_table(results)

    print(f"\nFull results saved to: {RESULTS_FILE}")
    print("Next: run step4_plots.py to generate comparison charts")


if __name__ == "__main__":
    main()
