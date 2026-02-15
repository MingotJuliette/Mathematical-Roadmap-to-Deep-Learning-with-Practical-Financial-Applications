import os
import csv
import time
from urllib.parse import urljoin

from utils.url import BASE, HISTORICAL, CALENDAR, SLEEP
from utils.fetchers import fetch
from utils.parsers import extract_title, extract_text
from utils.extractors import (
    extract_date_from_url,
    get_year_pages_2000_2019,
    get_proj_links_2006_2010,
    get_proj_links_2020_2025,
    get_press_conf_links, 
    get_proj_links_from_press_conf
)

PROJECTIONS_DIR = 'Deep_learning_gold_stock/data/raw/text/projections'
PROJECTIONS_METADATA_CSV = os.path.join(PROJECTIONS_DIR, "metadata.csv")

# projections_scraper.py
def ensure_output_projections():
    os.makedirs(PROJECTIONS_DIR, exist_ok=True)

def save_text_projections(item):
    filename = f"{item['date']}__{item['url'].split('/')[-1].replace('.htm','')}.txt"
    path = os.path.join(PROJECTIONS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(item["title"] + "\n\n")
        f.write(item["url"] + "\n\n")
        f.write(item["text"])

    return path


def append_metadata_projections(item):
    """Append metadata line to CSV."""
    header_needed = not os.path.exists(PROJECTIONS_METADATA_CSV)

    with open(PROJECTIONS_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "url", "type", "filepath"])
        if header_needed:
            writer.writeheader()
        writer.writerow({
            "date": item["date"],
            "url": item["url"],
            "type": "projection",
            "filepath": item["filepath"]
        })


# Main scraper function
def scrape_projections():

    print("\n===== SCRAPING FOMC PROJECTIONS =====\n")
    ensure_output_projections()

    # ------------------ 1) Historical 2006–2019 ------------------
    hist_html = fetch(HISTORICAL)
    if hist_html:
        year_pages = get_year_pages_2000_2019(hist_html)
        print(f"Found {len(year_pages)} historical year pages.")

        for yurl in year_pages:
            print(f"\nProcessing year page: {yurl}")
            year_html = fetch(yurl)
            if not year_html:
                continue

            # ---------- 2006–2010 ----------
            proj_0610 = get_proj_links_2006_2010(year_html)

            # ---------- 2011–2019 (two steps) ----------
            press_pages = get_press_conf_links(year_html)
            proj_1119 = []

            for purl in press_pages:
                phtml = fetch(purl)
                if not phtml:
                    continue

                proj_1119.extend(get_proj_links_from_press_conf(phtml))

            # Merge all projections for that year
            all_proj_links = sorted(set(proj_0610 + proj_1119))

            print(f"{yurl} → {len(all_proj_links)} projections")

            for url in all_proj_links:
                html = fetch(url)
                if not html:
                    continue

                item = {
                    "url": url,
                    "date": extract_date_from_url(url),
                    "title": extract_title(html),
                    "text": extract_text(html),
                }

                item["filepath"] = save_text_projections(item)
                append_metadata_projections(item)

                time.sleep(SLEEP)

    # ------------------ 2) Calendar 2020–2025 ------------------
    print("\nProcessing 2020–2025 calendar…")
    cal_html = fetch(CALENDAR)
    if cal_html:
        links = get_proj_links_2020_2025(cal_html)
        print(f"Calendar → {len(links)} projections")

        for url in links:
            html = fetch(url)
            if not html:
                continue

            item = {
                "url": url,
                "date": extract_date_from_url(url),
                "title": extract_title(html),
                "text": extract_text(html),
            }

            item["filepath"] = save_text_projections(item)
            append_metadata_projections(item)

            time.sleep(SLEEP)

    print("\n=== DONE scraping projections ===\n")
