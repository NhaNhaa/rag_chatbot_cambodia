"""
CSV Loader: convert structured data (CSV) into document chunks.
Uses a verified static dataset for Cambodian provinces (population, capital, area).
Source: National Institute of Statistics of Cambodia, 2024 estimates.
"""

import csv
import os
from typing import List, Dict

from config import DATA_FILE
from data_loader import load_documents_from_file, save_documents

PROVINCES_CSV = "data/provinces.csv"

# Verified data for all 25 provinces (Phnom Penh + 24 provinces)
# Populations based on official 2024 estimates (source: NIS)
PROVINCES_DATA = [
    {"name": "Phnom Penh", "capital": "Phnom Penh", "population": "2,353,000", "area_km2": "678"},
    {"name": "Banteay Meanchey", "capital": "Serei Saophoan", "population": "861,883", "area_km2": "6,679"},
    {"name": "Battambang", "capital": "Battambang", "population": "987,400", "area_km2": "11,702"},
    {"name": "Kampong Cham", "capital": "Kampong Cham", "population": "899,791", "area_km2": "4,549"},
    {"name": "Kampong Chhnang", "capital": "Kampong Chhnang", "population": "527,027", "area_km2": "5,521"},
    {"name": "Kampong Speu", "capital": "Kampong Speu", "population": "877,523", "area_km2": "7,017"},
    {"name": "Kampong Thom", "capital": "Kampong Thom", "population": "708,398", "area_km2": "13,814"},
    {"name": "Kampot", "capital": "Kampot", "population": "627,884", "area_km2": "4,873"},
    {"name": "Kandal", "capital": "Ta Khmau", "population": "1,265,805", "area_km2": "3,568"},
    {"name": "Kep", "capital": "Kep", "population": "41,798", "area_km2": "336"},
    {"name": "Koh Kong", "capital": "Koh Kong", "population": "123,618", "area_km2": "11,160"},
    {"name": "Kratié", "capital": "Kratié", "population": "374,755", "area_km2": "11,094"},
    {"name": "Mondulkiri", "capital": "Senmonorom", "population": "92,213", "area_km2": "14,288"},
    {"name": "Oddar Meanchey", "capital": "Samraong", "population": "276,038", "area_km2": "6,158"},
    {"name": "Pailin", "capital": "Pailin", "population": "79,445", "area_km2": "803"},
    {"name": "Preah Sihanouk", "capital": "Sihanoukville", "population": "310,072", "area_km2": "2,536"},
    {"name": "Preah Vihear", "capital": "Preah Vihear", "population": "254,827", "area_km2": "13,788"},
    {"name": "Prey Veng", "capital": "Prey Veng", "population": "1,146,000", "area_km2": "4,883"},
    {"name": "Pursat", "capital": "Pursat", "population": "419,952", "area_km2": "12,692"},
    {"name": "Ratanakiri", "capital": "Banlung", "population": "217,453", "area_km2": "10,782"},
    {"name": "Siem Reap", "capital": "Siem Reap", "population": "1,006,512", "area_km2": "10,299"},
    {"name": "Stung Treng", "capital": "Stung Treng", "population": "165,713", "area_km2": "11,092"},
    {"name": "Svay Rieng", "capital": "Svay Rieng", "population": "525,000", "area_km2": "2,966"},
    {"name": "Takeo", "capital": "Takeo", "population": "900,914", "area_km2": "3,563"},
    {"name": "Tboung Khmum", "capital": "Suong", "population": "776,841", "area_km2": "4,928"},
]

def ensure_csv_file():
    """Create provinces.csv using the verified static dataset."""
    if os.path.exists(PROVINCES_CSV):
        print(f"CSV file already exists: {PROVINCES_CSV}")
        # Overwrite to ensure correctness
        print("Overwriting with the corrected data.")
    os.makedirs(os.path.dirname(PROVINCES_CSV), exist_ok=True)
    with open(PROVINCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "capital", "population", "area_km2"])
        writer.writeheader()
        writer.writerows(PROVINCES_DATA)
    print(f"Created/updated {PROVINCES_CSV} with {len(PROVINCES_DATA)} provinces.")
    return True

def csv_to_documents(csv_path: str) -> List[Dict]:
    """Read CSV and convert each row into a document dict with enriched content."""
    docs = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "Unknown")
                capital = row.get("capital", "N/A")
                population = row.get("population", "N/A")
                area = row.get("area_km2", "N/A")
                content = (
                    f"{name} province has a population of {population}. "
                    f"Its capital city is {capital}. "
                    f"The area is {area} square kilometers. "
                    f"This is the province of {name} in Cambodia."
                )
                if "Phnom Penh" in name:
                    content += f" {name} is the province with the largest population in Cambodia."
                doc = {
                    "title": f"Province data: {name}",
                    "url": "",
                    "content": content,
                    "source_type": "csv",
                    "category": "geography"
                }
                docs.append(doc)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return docs

def main():
    print("=== CSV Loader ===\n")
    if not ensure_csv_file():
        return
    new_docs = csv_to_documents(PROVINCES_CSV)
    if not new_docs:
        print("No documents created from CSV. Check file format.")
        return

    existing_docs = load_documents_from_file(DATA_FILE)
    existing_titles = {doc["title"] for doc in existing_docs}
    added = 0
    for doc in new_docs:
        if doc["title"] not in existing_titles:
            existing_docs.append(doc)
            added += 1
            print(f"Added: {doc['title']}")
        else:
            print(f"Skipping duplicate: {doc['title']}")

    if added:
        save_documents(existing_docs, DATA_FILE)
        print(f"\nAdded {added} new province documents. Total: {len(existing_docs)}")
    else:
        print("No new province documents added.")

    print("\n⚠️ Run `python embedder.py` to rebuild the vectorstore.")

if __name__ == "__main__":
    main()