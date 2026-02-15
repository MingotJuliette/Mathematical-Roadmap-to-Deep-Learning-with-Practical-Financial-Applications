import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"

def extract_date_from_url(url):
    m = re.search(r"(20\d{2})(0\d|1[0-2])(0\d|[12]\d|3[01])", url)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "unknown"

#------------------------------------------------------------------------
# Year page
#------------------------------------------------------------------------

def get_year_pages_2000_2019(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # match only /monetarypolicy/fomchistorical2000–2019.htm
        if re.match(r"^/monetarypolicy/fomchistorical20(0\d|1\d)\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))


#------------------------------------------------------------------------
# Statement
#------------------------------------------------------------------------

# 2006–2010 and 2011–2019: extract statements from year pages
def get_statement_links(year_html):
    soup = BeautifulSoup(year_html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # 2000–2010 statements
        if re.match(r"^/newsevents/press/monetary/20\d{6}a\.htm$", href):
            links.append(urljoin(BASE, href))

        # 2011–today statements
        if re.match(r"^/newsevents/pressreleases/monetary20\d{6}a\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))

# --------------------------------------------------
# Extractors specific to projections
# --------------------------------------------------

# 2006–2010 projections
def get_proj_links_2006_2010(year_html):
    soup = BeautifulSoup(year_html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^/monetarypolicy/files/FOMC20\d{6}SEPmaterial\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))


# 2011–2019: first extract press conference pages
def get_press_conf_links(year_html):
    soup = BeautifulSoup(year_html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if re.match(r"^/monetarypolicy/fomcpresconf20\d{6}\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))


# 2011–2019: extract projections inside press conf pages
def get_proj_links_from_press_conf(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if re.match(r"^/monetarypolicy/fomcprojtabl20\d{6}\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))

# 2020–2025
def get_proj_links_2020_2025(calendar_html):
    soup = BeautifulSoup(calendar_html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if re.match(r"^/monetarypolicy/fomcprojtabl20\d{6}\.htm$", href):
            links.append(urljoin(BASE, href))

    return sorted(set(links))
