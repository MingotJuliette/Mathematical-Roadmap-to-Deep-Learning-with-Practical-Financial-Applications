
import time

from scrapers.statements_scraper import scrape_statements, STATEMENTS_DIR
from scrapers.projections_scraper import scrape_projections, PROJECTIONS_DIR

def main():
    """
    Main launcher for all FOMC scrapers:
      1. Statements
      3. Projections (SEP)
    Each scraper saves text files and updates its own metadata CSV.
    """
    start_time = time.time()
    print("\n===================================")
    print("STARTING FOMC SCRAPERS")
    print("===================================\n")

    # Scrape Statements
    print(">>> Scraping Statements ...\n")
    scrape_statements()
    print("\n>>> Finished Statements\n")
    time.sleep(1)

    # Scrape Projections (SEP)
    print(">>> Scraping Projections (SEP) ...\n")
    scrape_projections()
    print("\n>>> Finished Projections\n")
    time.sleep(1)

    end_time = time.time()
    elapsed = end_time - start_time
    print("\n===================================")
    print(f"ALL SCRAPERS FINISHED in {elapsed:.2f} seconds")
    print("===================================\n")


if __name__ == "__main__":
    main()
