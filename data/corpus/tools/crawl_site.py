# tools/crawl_site.py
# --- aspire ton site, nettoie le HTML, convertit en Markdown et écrit dans data/corpus/ ---

import os, re, time, hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from slugify import slugify
from tqdm import tqdm

START_URL = os.environ.get("START_URL", "https://pamelamagotte.fr/")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "300"))
OUT_DIR   = os.environ.get("OUT_DIR", "data/corpus")

# petites règles : reste dans le même domaine et ignore ces extensions
def same_host(url, base):
    return urlparse(url).netloc == urlparse(base).netloc

BAD_EXT = (".pdf",".jpg",".jpeg",".png",".gif",".webp",".svg",".zip",".mp3",".mp4",".avi",".mov",".wmv",".ico")

HEADERS = {"User-Agent": "PamelaBot/1.0 (+https://assistant-pamela.streamlit.app)"}

def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text
    except Exception:
        pass
    return None

def clean_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    # enlève scripts/styles/nav/footer/aside
    for tag in soup(["script","style","noscript","iframe"]):
        tag.decompose()
    for tag in soup.select("nav, footer, aside, form"):
        tag.decompose()

    # titre
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # prends l'article si présent, sinon le corps
    main = soup.select_one("article") or soup.select_one("main") or soup.body
    if not main:
        return title, ""

    # liens absolus
    for a in main.find_all("a", href=True):
        a["href"] = urljoin(base_url, a["href"])

    # convertit en Markdown
    text_md = md(str(main), strip=["img"])  # on ignore les images
    # nettoyage doux
    text_md = re.sub(r"\n{3,}", "\n\n", text_md).strip()

    return title, text_md

def write_markdown(title, url, body_md):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not title:
        # fallback : slug du chemin
        title = urlparse(url).path.strip("/").replace("/"," ") or "page"

    slug = slugify(title)[:80] or hashlib.sha1(url.encode()).hexdigest()[:10]
    fname = f"{slug}.md"
    path = os.path.join(OUT_DIR, fname)

    frontmatter = [
        "---",
        f'title: "{title.replace(\'"\', "\'")}"',
        f'url: "{url}"',
        'source: "website"',
        "---",
        ""
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(frontmatter))
        f.write(body_md.strip() + "\n")
    return path

def crawl():
    seen = set()
    queue = [START_URL]
    count = 0

    while queue and count < MAX_PAGES:
        url = queue.pop(0)
        if url in seen: 
            continue
        seen.add(url)

        if url.lower().endswith(BAD_EXT):
            continue
        if not same_host(url, START_URL):
            continue

        html = fetch(url)
        if not html:
            continue

        title, body_md = clean_html(html, url)
        if len(body_md) < 200:  # évite pages vides
            continue

        write_markdown(title, url, body_md)
        count += 1

        # trouve nouveaux liens
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            absu = urljoin(url, a["href"])
            if same_host(absu, START_URL) and absu not in seen:
                if not absu.lower().endswith(BAD_EXT):
                    queue.append(absu)

        # petite pause polie
        time.sleep(0.2)

    print(f"✅ Fini. Pages enregistrées dans {OUT_DIR} ({count} fichiers).")

if __name__ == "__main__":
    crawl()
