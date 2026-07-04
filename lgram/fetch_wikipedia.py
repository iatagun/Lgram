"""
Wikipedia corpus fetcher — multi-author modern expository texts.
Fetches ~30 articles on diverse modern topics for cross-validation.
"""

import json
import re
import urllib.request

TOPICS = [
    "Artificial_intelligence", "Climate_change", "Electric_vehicle",
    "Social_media", "Quantum_computing", "Renewable_energy",
    "CRISPR_gene_editing", "Blockchain", "Remote_work",
    "Mental_health", "Bitcoin", "Mars_2020",
    "mRNA_vaccine", "SpaceX", "Cybersecurity",
    "Gig_economy", "Urban_planning", "Data_privacy",
    "Renewable_energy", "Artificial_intelligence_in_healthcare",
    "Streaming_media", "Supply_chain", "Internet_of_things",
    "3D_printing", "Virtual_reality", "Genetic_testing",
    "Online_education", "Smart_city", "Drone_delivery",
    "Plant-based_meat", "Carbon_capture",
]

HEADERS = {"User-Agent": "centering-lgram/2.2.0 (research calibration)"}


def fetch_wikipedia_corpus(max_articles: int = 30) -> list[str]:
    """Fetch article intros from Wikipedia as modern expository corpus."""
    texts = []
    for topic in TOPICS[:max_articles]:
        try:
            url = (
                "https://en.wikipedia.org/w/api.php"
                "?action=query&prop=extracts&exintro&explaintext"
                f"&titles={topic}&format=json"
            )
            req = urllib.request.Request(url, headers=HEADERS)
            r = urllib.request.urlopen(req, timeout=15)
            data = json.loads(r.read())
            pages = data["query"]["pages"]
            for _pid, page in pages.items():
                text = page.get("extract", "")
                # clean references
                text = re.sub(r"\[[^\]]*\]", "", text)
                text = re.sub(r"\([^)]*\b(?:pronounced|known as|also called|Latin|French|German|Italian|Spanish|lit\.|abbreviation for|short for|formerly|born|died|fl\.)\s[^)]*\)", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text.split()) >= 50:
                    texts.append(text)
        except Exception as e:
            continue
    return texts


def save_corpus(filepath: str = "lgram/corpus_wikipedia.py"):
    """Fetch and save Wikipedia corpus as Python module."""
    texts = fetch_wikipedia_corpus(30)
    print(f"Fetched {len(texts)} Wikipedia articles")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write('"""Multi-author modern expository corpus — Wikipedia extracts."""\n\n')
        f.write("corpus_wikipedia = [\n")
        for t in texts:
            escaped = t.replace('"', '\\"').replace("\n", " ")
            f.write(f'    "{escaped}",\n')
        f.write("]\n")
    print(f"Saved to {filepath}")


if __name__ == "__main__":
    save_corpus()
