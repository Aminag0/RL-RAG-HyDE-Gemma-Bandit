"""
Step 1 - Data Collection
=========================
Paper: Assessing RAG and HyDE on 1B vs 4B Gemma LLMs (arxiv 2506.21568)

This script:
1. Downloads 300 physics papers from arXiv API
2. Generates synthetic personal data
3. Saves everything locally

Install:
    pip install requests arxiv pymupdf

Run:
    python step1_data_collection.py
"""

import os
import json
import time
import random
import requests

# ── Config ────────────────────────────────────────────────────────────────────
PAPERS_DIR   = "data/physics_papers"
PERSONAL_DIR = "data/personal"
META_FILE    = "data/papers_metadata.json"
NUM_PAPERS   = 300
DELAY        = 3.0   # seconds between requests — polite to arXiv

os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(PERSONAL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── Physics topics to fetch (matching paper's distribution) ───────────────────
PHYSICS_TOPICS = [
    "particle physics",
    "quantum field theory",
    "cosmology",
    "quantum mechanics",
    "condensed matter",
    "astrophysics",
    "thermodynamics",
    "electromagnetism",
    "nuclear physics",
    "optics",
]

PAPERS_PER_TOPIC = NUM_PAPERS // len(PHYSICS_TOPICS)   # 30 per topic


# ── Step 1A: Download arXiv papers ────────────────────────────────────────────
def fetch_arxiv_papers(query, max_results=30, start=0):
    """Fetch paper metadata from arXiv API."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def parse_arxiv_response(xml_text):
    """Parse arXiv API XML response into list of paper dicts."""
    import re
    papers = []

    entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    for entry in entries:
        def get_field(tag):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        arxiv_id_raw = get_field("id")
        arxiv_id = arxiv_id_raw.split("/abs/")[-1].strip()

        title    = re.sub(r"\s+", " ", get_field("title"))
        summary  = re.sub(r"\s+", " ", get_field("summary"))
        authors  = re.findall(r"<name>(.*?)</name>", entry)
        pub_date = get_field("published")[:10]

        if arxiv_id and title and summary:
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "summary":   summary,
                "authors":   authors[:3],
                "published": pub_date,
                "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}",
            })
    return papers


def save_paper_text(paper):
    """Save paper abstract + metadata as text file for ingestion."""
    safe_id  = paper["arxiv_id"].replace("/", "_")
    filepath = os.path.join(PAPERS_DIR, f"{safe_id}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {paper['title']}\n")
        f.write(f"AUTHORS: {', '.join(paper['authors'])}\n")
        f.write(f"DATE: {paper['published']}\n")
        f.write(f"ARXIV_ID: {paper['arxiv_id']}\n")
        f.write(f"PDF: {paper['pdf_url']}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(paper["summary"])

    return filepath


def collect_physics_papers():
    print("="*60)
    print("Collecting physics papers from arXiv")
    print("="*60)

    all_papers = []
    seen_ids   = set()

    for topic in PHYSICS_TOPICS:
        print(f"\nFetching: {topic} ({PAPERS_PER_TOPIC} papers)...")
        try:
            xml      = fetch_arxiv_papers(topic, max_results=PAPERS_PER_TOPIC)
            papers   = parse_arxiv_response(xml)
            new_papers = [p for p in papers if p["arxiv_id"] not in seen_ids]

            for paper in new_papers:
                seen_ids.add(paper["arxiv_id"])
                paper["topic"] = topic
                path = save_paper_text(paper)
                paper["filepath"] = path
                all_papers.append(paper)
                print(f"  Saved: {paper['arxiv_id']} — {paper['title'][:50]}")

            time.sleep(DELAY)

        except Exception as e:
            print(f"  Error fetching {topic}: {e}")
            continue

    # Save metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\nTotal papers collected: {len(all_papers)}")
    print(f"Saved to: {PAPERS_DIR}/")
    print(f"Metadata: {META_FILE}")
    return all_papers


# ── Step 1B: Generate synthetic personal data ─────────────────────────────────
# Paper used GPT-4 to generate synthetic data.
# We generate it directly — same approach.

PERSONAL_DATA = {
    "profile": {
        "name":        "Maryam Amjad",
        "age":         22,
        "email":       "maryam.amjad@example.com",
        "phone":       "+92-300-1234567",
        "address":     "House 45, Street 7, F-10/3, Islamabad, Pakistan",
        "occupation":  "Computer Science Student",
        "university":  "FAST-NUCES, Islamabad",
        "degree":      "Bachelor of Science in Computer Science",
        "grad_year":   2026,
        "gpa":         3.7,
    },
    "contacts": [
        {"name": "Dr. Qaiser Shafi",  "relation": "Supervisor",    "email": "qaiser@nu.edu.pk"},
        {"name": "Amna Nadeem",       "relation": "Teammate",      "email": "amna@example.com"},
        {"name": "Dur-e-Shahwar",     "relation": "Teammate",      "email": "dur@example.com"},
        {"name": "Ahmed Raza",        "relation": "Friend",        "email": "ahmed@example.com"},
        {"name": "Sara Khan",         "relation": "Classmate",     "email": "sara@example.com"},
    ],
    "schedule": [
        {"day": "Monday",    "time": "09:00", "event": "Machine Learning Lab",       "location": "CS Lab 3"},
        {"day": "Monday",    "time": "14:00", "event": "FYP Meeting with supervisor","location": "Faculty Room"},
        {"day": "Tuesday",   "time": "10:00", "event": "Data Structures Lecture",    "location": "LT-2"},
        {"day": "Wednesday", "time": "09:00", "event": "NLP Assignment submission",  "location": "Online"},
        {"day": "Thursday",  "time": "15:00", "event": "Study group session",        "location": "Library"},
        {"day": "Friday",    "time": "11:00", "event": "Research paper reading",     "location": "Home"},
    ],
    "preferences": {
        "programming_languages": ["Python", "Java", "C++"],
        "interests":             ["Machine Learning", "RAG systems", "Computer Vision"],
        "favorite_food":         "Biryani",
        "hobbies":               ["Reading", "Coding", "Table tennis"],
        "wake_up_time":          "07:00",
        "sleep_time":            "00:00",
    },
    "academic_history": [
        {"year": 2022, "semester": "Fall",   "gpa": 3.5, "courses": ["Programming Fundamentals", "Calculus", "Physics"]},
        {"year": 2023, "semester": "Spring", "gpa": 3.6, "courses": ["OOP", "Linear Algebra", "Digital Logic"]},
        {"year": 2023, "semester": "Fall",   "gpa": 3.7, "courses": ["Data Structures", "Database Systems", "OS"]},
        {"year": 2024, "semester": "Spring", "gpa": 3.8, "courses": ["Machine Learning", "Computer Networks", "AI"]},
        {"year": 2024, "semester": "Fall",   "gpa": 3.7, "courses": ["Deep Learning", "NLP", "Computer Vision"]},
    ],
    "projects": [
        {
            "name":        "Nexa Aegis",
            "description": "CSAM detection pipeline for Android forensics using YOLOv11 and Swin Transformer",
            "status":      "Completed",
            "year":        2025,
        },
        {
            "name":        "WHO Drug Alert RAG System",
            "description": "Multilingual RAG system over WHO counterfeit drug alerts using T-GRAG, Hyper-RAG, DELTA",
            "status":      "Completed",
            "year":        2025,
        },
    ],
    "conversation_history": [
        {"role": "user",      "message": "What is my GPA?",                   "timestamp": "2025-01-10"},
        {"role": "assistant", "message": "Your current GPA is 3.7",           "timestamp": "2025-01-10"},
        {"role": "user",      "message": "When is my next FYP meeting?",      "timestamp": "2025-01-11"},
        {"role": "assistant", "message": "Your FYP meeting is on Monday at 14:00 in the Faculty Room", "timestamp": "2025-01-11"},
    ],
}

# The 10 personal questions from paper evaluation
PERSONAL_QUESTIONS = [
    "What is my name and where do I study?",
    "What is my current GPA?",
    "Who is my FYP supervisor?",
    "What programming languages do I know?",
    "What are my hobbies and interests?",
    "What is my home address?",
    "When did I start my degree and when will I graduate?",
    "Who are my teammates on the project?",
    "What projects have I worked on?",
    "What is my Monday schedule?",
]

# The 12 physics questions from paper evaluation
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


def generate_personal_data():
    print("\nGenerating synthetic personal data...")

    # Save as JSON
    json_path = os.path.join(PERSONAL_DIR, "personal_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(PERSONAL_DATA, f, indent=2, ensure_ascii=False)

    # Save as flat text for RAG ingestion
    txt_path = os.path.join(PERSONAL_DIR, "personal_data.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        profile = PERSONAL_DATA["profile"]
        f.write(f"NAME: {profile['name']}\n")
        f.write(f"AGE: {profile['age']}\n")
        f.write(f"EMAIL: {profile['email']}\n")
        f.write(f"ADDRESS: {profile['address']}\n")
        f.write(f"UNIVERSITY: {profile['university']}\n")
        f.write(f"DEGREE: {profile['degree']}\n")
        f.write(f"GRADUATION YEAR: {profile['grad_year']}\n")
        f.write(f"GPA: {profile['gpa']}\n\n")

        f.write("CONTACTS:\n")
        for c in PERSONAL_DATA["contacts"]:
            f.write(f"  - {c['name']} ({c['relation']}): {c['email']}\n")

        f.write("\nSCHEDULE:\n")
        for s in PERSONAL_DATA["schedule"]:
            f.write(f"  - {s['day']} {s['time']}: {s['event']} at {s['location']}\n")

        f.write("\nPREFERENCES:\n")
        prefs = PERSONAL_DATA["preferences"]
        f.write(f"  Programming languages: {', '.join(prefs['programming_languages'])}\n")
        f.write(f"  Interests: {', '.join(prefs['interests'])}\n")
        f.write(f"  Hobbies: {', '.join(prefs['hobbies'])}\n")

        f.write("\nACADEMIC HISTORY:\n")
        for a in PERSONAL_DATA["academic_history"]:
            f.write(f"  - {a['year']} {a['semester']}: GPA {a['gpa']}, Courses: {', '.join(a['courses'])}\n")

        f.write("\nPROJECTS:\n")
        for p in PERSONAL_DATA["projects"]:
            f.write(f"  - {p['name']} ({p['year']}): {p['description']} [{p['status']}]\n")

    # Save questions
    q_path = os.path.join("data", "questions.json")
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump({
            "personal_questions": PERSONAL_QUESTIONS,
            "physics_questions":  PHYSICS_QUESTIONS,
        }, f, indent=2)

    print(f"  Personal data saved: {json_path}")
    print(f"  Personal data text:  {txt_path}")
    print(f"  Questions saved:     {q_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("RAG vs HyDE on Gemma — Data Collection")
    print("Paper: arxiv 2506.21568")
    print("="*60)

    # Generate personal data
    generate_personal_data()

    # Collect physics papers
    papers = collect_physics_papers()

    print("\n" + "="*60)
    print("Data collection complete.")
    print(f"  Physics papers : {len(papers)}")
    print(f"  Personal files : data/personal/")
    print(f"  Questions      : data/questions.json")
    print("\nNext: run step2_ingest.py to embed into Qdrant")
    print("="*60)


if __name__ == "__main__":
    main()
