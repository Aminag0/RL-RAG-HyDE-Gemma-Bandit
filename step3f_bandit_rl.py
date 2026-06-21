"""
Step 3F — RL Component: Epsilon-Greedy Multi-Armed Bandit
=========================================================

Purpose:
This file adds a real Reinforcement Learning component to the existing RAG/HyDE project.

RL Algorithm:
    Multi-Armed Bandit with Epsilon-Greedy action selection

Where RL is applied:
    Before retrieval, the agent chooses the retrieval policy/action:
        - RAG with k = 1, 3, 5, 7
        - HyDE with k = 1, 3, 5, 7

State:
    The current user query/question.

Actions / Arms:
    Each arm is a retrieval policy: (method, top_k)

Reward:
    Retrieval Quality Metric (RQM), i.e. mean cosine similarity between the original
    query embedding and the retrieved chunks.

Learning:
    The bandit updates the estimated value Q(a) of each retrieval action using:
        Q(a) <- Q(a) + (reward - Q(a)) / N(a)

Run:
    python step3f_bandit_rl.py
"""

import os
import json
import time
import random
import statistics
import requests
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
QDRANT_STORAGE  = "qdrant_storage"
COLLECTION_NAME = "physics_papers"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
MODEL           = "gemma3:1b"
TIMEOUT         = 120

RESULTS_FILE    = "data/results_bandit_rl.json"
PLOTS_DIR       = "data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

EPSILON         = 0.20     # 20% exploration, 80% exploitation
EPISODES        = 5        # repeat question set 5 times so the bandit can learn
K_VALUES        = [1, 3, 5, 7]
RANDOM_SEED     = 42

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


# ── Helper functions ──────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def generate_hypothetical_doc(question):
    """HyDE step: generate hypothetical document for retrieval."""
    prompt = f"""Write a brief scientific passage answering: "{question}"
Write as if from a physics textbook. Keep under 80 words."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 120,
                    "num_ctx": 512,
                },
            },
            timeout=TIMEOUT,
        )
        return response.json().get("response", "").strip()
    except Exception:
        # Fallback keeps the script runnable even if Ollama is unavailable.
        return question


def retrieve_chunks(query_vec, qdrant, top_k):
    """Retrieve top-k chunks from Qdrant."""
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points


def compute_rqm(original_query_vec, retrieved_chunks, embedder):
    """
    Reward function: Retrieval Quality Metric.
    Mean cosine similarity between original query and retrieved chunks.
    """
    if not retrieved_chunks:
        return 0.0

    scores = []
    for chunk in retrieved_chunks:
        chunk_vec = embedder.encode(chunk)
        scores.append(cosine_similarity(original_query_vec, chunk_vec))

    return float(statistics.mean(scores))


@dataclass
class BanditArm:
    method: str
    k: int
    q_value: float = 0.0
    count: int = 0

    @property
    def name(self):
        return f"{self.method}_k{self.k}"


class EpsilonGreedyBandit:
    """
    Multi-Armed Bandit using epsilon-greedy exploration.

    With probability epsilon:
        explore by selecting a random retrieval policy.

    With probability 1 - epsilon:
        exploit by selecting the retrieval policy with highest learned Q-value.
    """
    def __init__(self, arms, epsilon=0.2):
        self.arms = arms
        self.epsilon = epsilon

    def select_action(self):
        if random.random() < self.epsilon:
            return random.choice(self.arms), "explore"

        max_q = max(arm.q_value for arm in self.arms)
        best_arms = [arm for arm in self.arms if arm.q_value == max_q]
        return random.choice(best_arms), "exploit"

    def update(self, arm, reward):
        arm.count += 1
        arm.q_value = arm.q_value + (reward - arm.q_value) / arm.count


# ── RL Experiment ─────────────────────────────────────────────────────────────
def run_bandit_rl():
    print("=" * 70)
    print("Step 3F — RL Component: Epsilon-Greedy Multi-Armed Bandit")
    print("=" * 70)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n[1/3] Loading models and Qdrant...")
    embedder = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(path=QDRANT_STORAGE)
    print(f"   Embedding model: {EMBED_MODEL}")
    print(f"   Qdrant vectors : {qdrant.count(COLLECTION_NAME).count}")

    arms = []
    for method in ["RAG", "HyDE"]:
        for k in K_VALUES:
            arms.append(BanditArm(method=method, k=k))

    bandit = EpsilonGreedyBandit(arms, epsilon=EPSILON)

    print("\n[2/3] Running RL bandit episodes...")
    history = []

    for episode in range(1, EPISODES + 1):
        print(f"\nEpisode {episode}/{EPISODES}")

        for question_num, question in enumerate(PHYSICS_QUESTIONS, start=1):
            start_time = time.time()

            # State = current question
            query_vec = embedder.encode(question)

            # Action selection by RL agent
            arm, decision_type = bandit.select_action()

            # Execute selected action / retrieval policy
            if arm.method == "RAG":
                retrieval_query_vec = query_vec
                hypothetical_doc = None
            else:
                hypothetical_doc = generate_hypothetical_doc(question)
                retrieval_query_vec = embedder.encode(hypothetical_doc)

            retrieved = retrieve_chunks(retrieval_query_vec.tolist(), qdrant, arm.k)
            chunks = [r.payload.get("text", "") for r in retrieved]

            # Reward = RQM measured against ORIGINAL query
            reward = compute_rqm(query_vec, chunks, embedder)

            # RL update
            bandit.update(arm, reward)

            elapsed = time.time() - start_time

            row = {
                "episode": episode,
                "question_num": question_num,
                "question": question,
                "selected_arm": arm.name,
                "method": arm.method,
                "k": arm.k,
                "decision_type": decision_type,
                "reward_rqm": reward,
                "updated_q_value": arm.q_value,
                "arm_count": arm.count,
                "time_seconds": elapsed,
                "hypothetical_doc": hypothetical_doc,
            }
            history.append(row)

            print(
                f"   Q{question_num:02d} | {decision_type:<7} | "
                f"action={arm.name:<8} | reward={reward:.4f} | "
                f"Q={arm.q_value:.4f} | N={arm.count}"
            )

    print("\n[3/3] Saving results...")

    final_policy = sorted(
        [
            {
                "arm": arm.name,
                "method": arm.method,
                "k": arm.k,
                "q_value": arm.q_value,
                "selected_count": arm.count,
            }
            for arm in arms
        ],
        key=lambda x: x["q_value"],
        reverse=True,
    )

    best_arm = final_policy[0]

    summary = {
        "algorithm": "Epsilon-Greedy Multi-Armed Bandit",
        "epsilon": EPSILON,
        "episodes": EPISODES,
        "state": "Current physics question/query",
        "actions": [arm.name for arm in arms],
        "reward": "Retrieval Quality Metric (mean cosine similarity between original query and retrieved chunks)",
        "update_rule": "Q(a) <- Q(a) + (reward - Q(a)) / N(a)",
        "best_learned_policy": best_arm,
        "final_policy_ranking": final_policy,
        "history": history,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Results saved: {RESULTS_FILE}")
    print("\nFinal learned policy ranking:")
    for i, row in enumerate(final_policy, start=1):
        print(
            f"   {i}. {row['arm']:<8} | Q={row['q_value']:.4f} | "
            f"selected={row['selected_count']} times"
        )

    print(f"\nBest learned retrieval policy: {best_arm['arm']}")

    return summary


def plot_bandit_results(summary):
    final_policy = summary["final_policy_ranking"]

    arms = [row["arm"] for row in final_policy]
    q_values = [row["q_value"] for row in final_policy]
    counts = [row["selected_count"] for row in final_policy]

    # Plot 1: learned Q-values
    plt.figure(figsize=(10, 5))
    plt.bar(arms, q_values)
    plt.xlabel("Retrieval Policy / Bandit Arm")
    plt.ylabel("Learned Q-value (Expected RQM Reward)")
    plt.title("Epsilon-Greedy Bandit: Learned Retrieval Policy Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "figure_bandit_q_values.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: action selection counts
    plt.figure(figsize=(10, 5))
    plt.bar(arms, counts)
    plt.xlabel("Retrieval Policy / Bandit Arm")
    plt.ylabel("Number of Times Selected")
    plt.title("Epsilon-Greedy Bandit: Exploration vs Exploitation Selections")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "figure_bandit_action_counts.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Saved plot: {path1}")
    print(f"   Saved plot: {path2}")


if __name__ == "__main__":
    summary = run_bandit_rl()
    plot_bandit_results(summary)
