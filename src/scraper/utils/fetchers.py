import requests
import time

UA = {"User-Agent": "fomc-scraper/1.0"}

def fetch(url, sleep=0.3):
    print(f"[FETCH] {url}")
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        time.sleep(sleep)
        return r.text
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None
