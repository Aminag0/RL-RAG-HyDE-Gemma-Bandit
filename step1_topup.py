"""
Top-up script — downloads remaining papers to reach 300 total
Run after step1_data_collection.py
    python step1_topup.py
"""

import os
import json
import time
import requests
import re

PAPERS_DIR = "data/physics_papers"
META_FILE  = "data/papers_metadata.json"
TARGET     = 300
DELAY      = 4.0

os.makedirs(PAPERS_DIR, exist_ok=True)

# More specific queries to get fresh results
EXTRA_QUERIES = [
    "quantum gravity",
    "string theory",
    "plasma physics",
    "statistical mechanics",
    "fluid dynamics",
    "solid state physics",
    "biophysics",
    "geophysics",
    "photonics",
    "superconductivity",
    "relativity spacetime",
    "neutrino physics",
    "gravitational waves",
    "quantum computing physics",
    "chaos theory physics",
]


def fetch_arxiv(query, max_results=15, start=0):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def parse_papers(xml_text):
    papers = []
    entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    for entry in entries:
        def get(tag):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        arxiv_id = get("id").split("/abs/")[-1].strip()
        title    = re.sub(r"\s+", " ", get("title"))
        summary  = re.sub(r"\s+", " ", get("summary"))
        authors  = re.findall(r"<name>(.*?)</name>", entry)
        date     = get("published")[:10]

        if arxiv_id and title and summary:
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "summary":   summary,
                "authors":   authors[:3],
                "published": date,
                "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}",
            })
    return papers


def save_paper(paper):
    safe_id  = paper["arxiv_id"].replace("/", "_")
    filepath = os.path.join(PAPERS_DIR, f"{safe_id}.txt")
    if os.path.exists(filepath):
        return None
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {paper['title']}\n")
        f.write(f"AUTHORS: {', '.join(paper['authors'])}\n")
        f.write(f"DATE: {paper['published']}\n")
        f.write(f"ARXIV_ID: {paper['arxiv_id']}\n")
        f.write(f"PDF: {paper['pdf_url']}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(paper["summary"])
    return filepath


def main():
    # Count existing
    existing = set(f.replace(".txt","") for f in os.listdir(PAPERS_DIR) if f.endswith(".txt"))
    current  = len(existing)
    needed   = TARGET - current

    print(f"Current papers : {current}")
    print(f"Target         : {TARGET}")
    print(f"Need to fetch  : {needed}")

    if needed <= 0:
        print("Already at 300. Nothing to do.")
        return

    # Load existing metadata
    all_papers = []
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            all_papers = json.load(f)
    seen_ids = set(p["arxiv_id"] for p in all_papers)

    new_count = 0
    for query in EXTRA_QUERIES:
        if new_count >= needed:
            break

        print(f"\nFetching: {query}...")
        try:
            xml    = fetch_arxiv(query, max_results=15)
            papers = parse_papers(xml)

            for paper in papers:
                if new_count >= needed:
                    break
                if paper["arxiv_id"] in seen_ids:
                    continue

                path = save_paper(paper)
                if path:
                    paper["filepath"] = path
                    all_papers.append(paper)
                    seen_ids.add(paper["arxiv_id"])
                    new_count += 1
                    print(f"  [{current + new_count}/{TARGET}] {paper['arxiv_id']} — {paper['title'][:50]}")

            time.sleep(DELAY)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)
            continue

    # Save updated metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    final = len(os.listdir(PAPERS_DIR))
    print(f"\nDone. Total papers: {final}")


if __name__ == "__main__":
    main()
