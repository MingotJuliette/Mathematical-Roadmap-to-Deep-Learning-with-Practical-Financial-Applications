# scrapers/statements_scraper.py

import os
import csv
import time
from urllib.parse import urljoin

from utils.url import BASE, HISTORICAL, CALENDAR, SLEEP
from utils.fetchers import fetch
from utils.parsers import extract_title, extract_text
from utils.extractors import (
    extract_date_from_url,
    get_statement_links,
    get_year_pages_2000_2019
)

STATEMENTS_DIR = 'data/raw/text/statements'
STATEMENTS_METADATA_CSV = os.path.join(STATEMENTS_DIR, "metadata.csv")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

# statements_scraper.py
def ensure_output_statements():
    os.makedirs(STATEMENTS_DIR, exist_ok=True)

def save_text_statements(item):
    filename = f"{item['date']}__{item['url'].split('/')[-1].replace('.htm','')}.txt"
    path = os.path.join(STATEMENTS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(item["title"] + "\n\n")
        f.write(item["url"] + "\n\n")
        f.write(item["text"])

    return path


def append_metadata_statements(item):
    """Append metadata line to CSV."""
    header_needed = not os.path.exists(STATEMENTS_METADATA_CSV)

    with open(STATEMENTS_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "url", "type", "filepath"])
        if header_needed:
            writer.writeheader()
        writer.writerow({
            "date": item["date"],
            "url": item["url"],
            "type": "statement",
            "filepath": item["filepath"]
        })

# --------------------------------------------------
# Main scraper function
# --------------------------------------------------

def scrape_statements():

    print("\n===== SCRAPING FOMC STATEMENTS =====\n")
    ensure_output_statements()

    # ------------------ 1) Historical statements 2006–2019 ------------------
    hist_html = fetch(HISTORICAL)
    if hist_html:
        year_pages = get_year_pages_2000_2019(hist_html)
        print(f"Found {len(year_pages)} historical year pages.")

        for yurl in year_pages:
            year_html = fetch(yurl)
            if not year_html:
                continue

            links = get_statement_links(year_html)
            print(f"{yurl} → {len(links)} statements")

            for url in links:
                html = fetch(url)
                if not html:
                    continue

                item = {
                    "url": url,
                    "date": extract_date_from_url(url),
                    "title": extract_title(html),
                    "text": extract_text(html)
                }
                item["filepath"] = save_text_statements(item)
                append_metadata_statements({
                    **item,
                    "type": "statement"  # add type here for CSV
                })
                time.sleep(SLEEP)

    # ------------------ 2) Calendar statements 2020–2025 ------------------
    cal_html = fetch(CALENDAR)
    if cal_html:
        links = get_statement_links(cal_html)  # same extractor works with regex
        print(f"\nCalendar 2020–2025 → {len(links)} statements\n")

        for url in links:
            html = fetch(url)
            if not html:
                continue

            item = {
                "url": url,
                "date": extract_date_from_url(url),
                "title": extract_title(html),
                "text": extract_text(html)
            }
            item["filepath"] = save_text_statements(item)
            append_metadata_statements({
                **item,
                "type": "statement"
            })
            time.sleep(SLEEP)

    print("\n=== DONE scraping statements ===\n")
