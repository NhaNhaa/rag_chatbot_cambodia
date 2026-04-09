"""
Advanced data loader: fetch more Wikipedia pages, add synthetic FAQ entries,
and merge with existing documents.json.
Now fetches "Cambodian cuisine" from Wikipedia (no hardcoded data).
"""

import json
import time
from typing import List, Dict

from data_loader import fetch_wikipedia_page, load_documents_from_file, save_documents
from config import DATA_FILE

# Additional Wikipedia topics to fetch (including those that failed earlier)
ADDITIONAL_WIKI_TOPICS = [
    "Preah Vihear Temple",
    "Bayon",
    "Tonlé Sap",
    "Kampot (city)",
    "Cambodian genocide",
    "Khmer Rouge",
    "Mekong river",
    "Cardamom Mountains",
    "Apsara dance",
    "Pchum Ben",
    "Khmer Empire",
    "Geography of Cambodia",
    "Angkor",
    "Cambodian cuisine",   # <-- add this to fetch real Wikipedia article
]

# Synthetic FAQ entries (only travel FAQ; cuisine is now fetched from Wikipedia)
SYNTHETIC_FAQ = [
    {
        "title": "Cambodia Travel FAQ",
        "url": "",
        "content": """Q: What is the best time to visit Cambodia?
A: The best time is from November to February when the weather is cool and dry.

Q: Do I need a visa?
A: Yes, most tourists need a visa. You can get an e-visa online or a visa on arrival.

Q: What currency is used?
A: Cambodian Riel (KHR) and US dollars are widely accepted.

Q: Is it safe to travel?
A: Cambodia is generally safe, but be cautious of petty theft in crowded areas.

Q: What language is spoken?
A: Khmer is the official language; English is common in tourist areas.

Q: What are the must-see temples?
A: Angkor Wat, Bayon, Ta Prohm, Banteay Srei, and Preah Vihear."""
    }
    # Removed the hardcoded cuisine FAQ – will fetch from Wikipedia instead
]

def fetch_new_wiki_documents(existing_titles: set) -> List[Dict]:
    """Fetch Wikipedia pages not already in the knowledge base."""
    new_docs = []
    for topic in ADDITIONAL_WIKI_TOPICS:
        if topic in existing_titles:
            print(f"Skipping {topic} – already exists.")
            continue
        print(f"Fetching: {topic}")
        doc = fetch_wikipedia_page(topic)
        if doc and doc.get("content"):
            new_docs.append(doc)
            print(f"  ✓ Added {topic}")
        else:
            print(f"  ✗ Failed to fetch {topic}")
        time.sleep(0.5)
    return new_docs

def add_synthetic_docs(existing_titles: set) -> List[Dict]:
    """Add synthetic FAQ entries if not already present."""
    new_docs = []
    for faq in SYNTHETIC_FAQ:
        title = faq["title"]
        if title in existing_titles:
            print(f"Skipping synthetic {title} – already exists.")
            continue
        new_docs.append(faq)
        print(f"  ✓ Added synthetic: {title}")
    return new_docs

def main():
    print("=== Advanced Data Loader ===\n")
    existing_docs = load_documents_from_file(DATA_FILE)
    existing_titles = {doc["title"] for doc in existing_docs}
    print(f"Existing documents: {len(existing_docs)}")

    new_wiki = fetch_new_wiki_documents(existing_titles)
    new_synthetic = add_synthetic_docs(existing_titles)

    all_new = new_wiki + new_synthetic
    if all_new:
        merged = existing_docs + all_new
        save_documents(merged, DATA_FILE)
        print(f"\nAdded {len(all_new)} new documents. Total: {len(merged)}")
    else:
        print("No new documents added.")

    print("\n⚠️ Run `python embedder.py` to rebuild the vectorstore.")

if __name__ == "__main__":
    main()