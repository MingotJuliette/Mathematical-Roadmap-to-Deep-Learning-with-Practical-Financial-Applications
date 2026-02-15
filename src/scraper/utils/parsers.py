from bs4 import BeautifulSoup

def extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "Untitled"

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    art = soup.find("article")
    if art:
        return art.get_text("\n", strip=True)
    return "\n\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
