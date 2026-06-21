\# RAG vs HyDE on Compact Gemma LLMs with RL Bandit Extension



\## Project Overview

This project implements and extends a RAG vs HyDE evaluation on compact Gemma LLMs using local inference with Ollama, Qdrant vector search, and MongoDB.



\## Reinforcement Learning Component



A real RL component has been added in:



`step3f\_bandit\_rl.py`



\## RL Algorithm

Epsilon-Greedy Multi-Armed Bandit



\## Where RL is Applied

The RL agent is applied at the retrieval policy selection stage.



\## State

Current user query / physics question.



\## Actions / Arms

\- RAG\_k1

\- RAG\_k3

\- RAG\_k5

\- RAG\_k7

\- HyDE\_k1

\- HyDE\_k3

\- HyDE\_k5

\- HyDE\_k7



\## Reward

Retrieval Quality Metric (RQM), calculated as the mean cosine similarity between the original query and retrieved chunks.



\## Update Rule

Q(a) = Q(a) + (reward - Q(a)) / N(a)



\## RL Outputs

\- `data/results\_bandit\_rl.json`

\- `data/plots/figure\_bandit\_q\_values.png`

\- `data/plots/figure\_bandit\_action\_counts.png`



\## Main Files

\- `step1b\_full\_pipeline.py` — data ingestion and Qdrant setup

\- `step3\_evaluation.py` — Baseline, RAG, and HyDE evaluation

\- `step3d\_retrieval\_quality.py` — RQM analysis

\- `step3e\_topk\_analysis.py` — Top-K sensitivity analysis

\- `step3f\_bandit\_rl.py` — RL bandit component

\- `step4\_plots.py` — plot generation

