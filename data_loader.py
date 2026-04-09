"""
Fetch Wikipedia articles about Cambodia tourism topics.
Saves raw text as JSON documents for later chunking and embedding.
Now includes a function to fetch the full page content (not just extract).
"""

import json
import time
import re
from typing import List, Dict

import requests

from config import DATA_FILE


# List of tourism topics (page titles) to fetch
TOPICS = [
    "Angkor Wat",
    "Phnom Penh",
    "Siem Reap",
    "Cambodian cuisine",
    "Bayon Temple",
    "Tonle Sap",
    "Royal Palace of Cambodia",
    "History of Cambodia",
    "Preah Vihear Temple",
    "Kampot (town)",
]

# Wikipedia requires a User-Agent header (identify your app)
HEADERS = {
    "User-Agent": "CambodiaTourismRAGBot/1.0 (https://github.com/yourusername/rag_chatbot_cambodia; your.email@example.com) Educational Project"
}


def fetch_wikipedia_page(title: str) -> Dict[str, str]:
    """
    Fetch a single Wikipedia page by title (returns extract, may be short).
    Returns dict with 'title', 'url', 'content' (extracted text).
    Returns empty dict if fetch fails.
    """
    print(f"Fetching: {title}")

    # Wikipedia API endpoint
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exintro": False,          # get full article, not just intro
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extract page content from the response
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if page_id == "-1":
                # Page not found
                print(f"  Warning: '{title}' not found on Wikipedia")
                return {}
            return {
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "content": page_info.get("extract", "").strip(),
            }

        return {}

    except requests.exceptions.RequestException as error:
        print(f"  Network error fetching {title}: {error}")
        return {}
    except json.JSONDecodeError as error:
        print(f"  JSON error for {title}: {error}")
        return {}
    except Exception as error:
        print(f"  Unexpected error for {title}: {error}")
        return {}


def fetch_full_wikipedia_page(title: str) -> Dict[str, str]:
    """
    Fetch the full page content of a Wikipedia article using the 'parse' API.
    Returns dict with 'title', 'url', 'content' (full text, cleaned from HTML).
    Returns empty dict if fetch fails.
    """
    print(f"Fetching full page: {title}")
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "formatversion": "2",
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        html = data.get("parse", {}).get("text", {})
        if not html:
            return {}
        # Simple HTML to plain text conversion (remove tags)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "content": text,
        }
    except Exception as error:
        print(f"Error fetching full page for {title}: {error}")
        return {}


def fetch_all_topics(topics: List[str]) -> List[Dict[str, str]]:
    """
    Fetch all Wikipedia pages for the given list of topics.
    Adds a small delay between requests to be polite to Wikipedia.
    Returns list of document dicts (skips failed ones).
    """
    documents = []
    for topic in topics:
        doc = fetch_wikipedia_page(topic)
        if doc and doc.get("content"):
            documents.append(doc)
            print(f"  ✓ Added: {topic} ({len(doc['content'])} chars)")
        else:
            print(f"  ✗ Skipped: {topic} (no content)")
        # Be respectful to Wikipedia API
        time.sleep(0.5)
    return documents


def save_documents(documents: List[Dict[str, str]], output_path: str) -> bool:
    """
    Save documents list to a JSON file.
    Returns True if successful, False otherwise.
    """
    if not documents:
        print("No documents to save.")
        return False

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(documents, file, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(documents)} documents to {output_path}")
        return True
    except IOError as error:
        print(f"Error writing to {output_path}: {error}")
        return False


def load_documents_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    Load previously saved documents from JSON file.
    Returns empty list if file doesn't exist or is corrupted.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        else:
            print(f"Warning: {file_path} does not contain a list.")
            return []
    except FileNotFoundError:
        print(f"No existing file at {file_path}. Will create new one.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {file_path} is corrupted. Starting fresh.")
        return []
    except Exception as error:
        print(f"Unexpected error loading {file_path}: {error}")
        return []


# ========== Main execution (when run directly) ==========
if __name__ == "__main__":
    print("=== Cambodia Tourism Data Loader ===\n")

    # Check if we already have data (optional: skip download)
    existing = load_documents_from_file(DATA_FILE)
    if existing:
        print(f"Found {len(existing)} existing documents in {DATA_FILE}")
        answer = input("Download fresh data anyway? (y/N): ").strip().lower()
        if answer != "y":
            print("Keeping existing data. Exiting.")
            exit(0)

    # Fetch new data
    print(f"Fetching {len(TOPICS)} Wikipedia articles...\n")
    docs = fetch_all_topics(TOPICS)

    if docs:
        save_documents(docs, DATA_FILE)
        print(f"\nSuccess! Downloaded {len(docs)} articles.")
    else:
        print("\nNo documents were fetched. Check your internet connection.")