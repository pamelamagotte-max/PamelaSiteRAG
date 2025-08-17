# app.py — Assistant Pamela (corpus auto .md OU .odt/.pdf, anti-RateLimit, cache disque)
import os, re, csv, glob, json, time, hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
from openai import OpenAI

# ================ Secrets & config ================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
ADMIN_CODE     = st.secrets.get("ADMIN_CODE", "pam2025")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY n'est pas configurée dans les Secrets Streamlit.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

DATA_DIR   = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"   # .md (avec ou sans front-matter)
DOCS_DIR   = DATA_DIR / "docs"     # .odt/.pdf/.txt/.md/.html
EMB_FILE   = DATA_DIR / "embeddings.npy"
EMB_META   = DATA_DIR / "embeddings_meta.json"
LOG_PATH   = DATA_DIR / "chat_logs.csv"

for d in [DATA_DIR, CORPUS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================ Identité & style ================
SYSTEM_PROMPT = (
    "Tu es l'assistant de Pamela Magotte. Voix claire, bienveillante et structurée. "
    "Tu relies les sujets à la pensée systémique, aux patterns (motifs récurrents) et à la pleine conscience. "
    "Tu simplifies sans appauvrir, donnes des exemples concrets et des mini-exercices. "
    "Si une question sort de ton périmètre, tu le dis et proposes une reformulation utile."
)

# ================ Fallback mini-corpus ================
FALLBACK = [
    {"source": "10_piliers_intro",    "title":"Les 10 piliers – introduction", "url":"", "text":
     "Les 10 piliers proposent une grille systémique pour relier perception, action et responsabilité."},
    {"source": "pleine_conscience",   "title":"Pleine conscience – bases",     "url":"", "text":
     "La pleine conscience aide à voir les boucles automatiques et à choisir une réponse plutôt qu'une réaction."},
    {"source": "management_patterns", "title":"Management & patterns",          "url":"", "text":
     "Repérer les patterns permet d'intervenir au bon niveau du système : structure, règles, langage, rituels, feedbacks."},
]

# ================ Lecture .md (corpus/) ================
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
def parse_frontmatter(raw: str) -> Tuple[Dict, str]:
    meta, body = {}, raw
    m = FM_RE.match(raw)
    if m:
        fm = m.group(1)
        body = raw[m.end():]
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body.strip()

def load_corpus_md(max_files=2000, max_chars=12000) -> List[Dict]:
    paths = sorted(glob.glob(str(CORPUS_DIR / "*.md")))
    docs = []
    for p in paths[:max_files]:
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        meta, body = parse_frontmatter(raw)
        title = meta.get("title") or Path(p).stem
        url   = meta.get("url", "")
        text  = body.strip()[:max_chars]
        if len(text) < 200:
            continue
        docs.append({"source": Path(p).name, "title": title, "url": url, "text": text})
    return docs

# ================ Lecture docs (docs/ : odt/pdf/txt/md/html) ================
def _clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_txt(path: Path) -> str:
    return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def read_md_file(path: Path) -> str:
    return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md
    except Exception:
        return ""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = isinstance(html, str) and BeautifulSoup(html, "html.parser")
    if not soup:
        return ""
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script","style","noscript","iframe"]): tag.decompose()
    body = md(str(soup), strip=["img"])
    return _clean_text(f"# {title}\n\n{body}")

def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    txt = []
    with path.open("rb") as f:
        try:
            pdf = PdfReader(f)
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                txt.append(t)
        except Exception:
            return ""
    return _clean_text("\n\n".join(txt))

def read_odt(path: Path) -> str:
    try:
        from odf.opendocument import load as odf_load
        from odf import text as odf_text, teletype as odf_teletype
    except Exception:
        return ""
    try:
        doc = odf_load(str(path))
        paras = doc.getElementsByType(odf_text.P)
        txt = "\n".join(odf_teletype.extractText(p) for p in paras)
        return _clean_text(txt)
    except Exception:
        return ""

ALLOWED = {".odt", ".pdf", ".txt", ".md", ".html", ".htm"}
def load_docs_dir(max_files=1000, max_chars=12000) -> List[Dict]:
    files = [p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED]
    files = sorted(files)[:max_files]
    out = []
    for p in files:
        ext = p.suffix.lower()
        if ext == ".odt":  text = read_odt(p)
        elif ext == ".pdf": text = read_pdf(p)
        elif ext == ".txt": text = read_txt(p)
        elif ext in {".html",".htm"}: text = read_html(p)
        elif ext == ".md": text = read_md_file(p)
        else: text = ""
        text = text[:max_chars]
        if text and len(text) >= 200:
            out.append({"source": str(p.relative_to(DOCS_DIR)), "title": p.stem, "url": "", "text": text})
    return out

def load_documents() -> Tuple[List[Dict], str]:
    docs_md = load_corpus_md()
    if docs_md:
        return docs_md, "md"
    docs_fs = load_docs_dir()
    if docs_fs:
        return docs_fs, "files"
    return FALLBACK, "fallback"

DOCS, CORPUS_MODE = load_documents()

# ================ Embeddings robustes (lots + retry + cache) ================
EMB_MODEL = "text-embedding-3-small"

def files_signature() -> str:
    """Empreinte: tailles+mtimes des fichiers (md ou docs) pour invalider le cache au bon moment."""
    parts = []
    roots = [CORPUS_DIR] if CORPUS_MODE == "md" else [DOCS_DIR]
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                try:
                    parts.append(f"{p}:{p.stat().st_size}:{int(p.stat().st_mtime)}")
                except Exception:
                    pass
    if not parts and CORPUS_MODE == "fallback":
        parts = [f"fallback:{len(DOCS)}"]
    h = hashlib.md5("|".join(parts).encode()).hexdigest()
    return h

def backoff_sleep(attempt: int):
    time.sleep(min(8, 0.5 * (2 ** attempt)))  # 0.5,1,2,4,8,8...

def embed_in_batches(texts: List[str], batch: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        for attempt in range(6):
            try:
                resp = client.embeddings.create(model=EMB_MODEL, input=chunk)
                vecs.extend([d.embedding for d in resp.data])
                break
            except Exception:
                if attempt == 5:
                    raise
                backoff_sleep(attempt)
        time.sleep(0.2)  # politesse
    return np.array(vecs, dtype=float)

@st.cache_data(show_spinner=True)
def build_or_load_embeddings(sig: str) -> Tuple[np.ndarray, int]:
    # Si fichiers & meta présents et signature identique → charge
    if EMB_FILE.exists() and EMB_META.exists():
        try:
            meta = json.loads(EMB_META.read_text(encoding="utf-8"))
            if meta.get("sig") == sig and meta.get("n") == len(DOCS):
                M = np.load(EMB_FILE)
                return M, len(DOCS)
        except Exception:
            pass
    texts = [d["text"] for d in DOCS]
    # Option: limiter si énorme corpus, puis augmenter progressivement
    # texts = texts[:600]
    M = embed_in_batches(texts, batch=64)
    try:
        np.save(EMB_FILE, M)
        EMB_META.write_text(json.dumps({"sig": sig, "n": len(DOCS)}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return M, len(DOCS)

def embed_query(q: str) -> np.ndarray:
    for attempt in range(6):
        try:
            r = client.embeddings.create(model=EMB_MODEL, input=[q])
            return np.array(r.data[0].embedding, dtype=float)
        except Exception:
            if attempt == 5:
                raise
            backoff_sleep(attempt)

def top_k_docs(query: str, k: int, M: np.ndarray) -> List[Dict]:
    if M.shape[0] == 0:
        return []
    qv = embed_query(query)
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(qv) + 1e-9))
    idx = np.argsort(-sims)[:k]
    return [DOCS[i] for i in idx]

# ================ Log ================
def log_chat(email, question, answer):
    newfile = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["timestamp", "email", "question", "answer"])
        w.writerow([datetime.utcnow().isoformat(), email or "", question, answer])

# ================ UI ================
st.set_page_config(page_title="Assistant Pamela", page_icon="💬", layout="centered")
st.title("💬 Assistant Pamela")

admin_val = st.query_params.get("admin", "")
is_admin  = (admin_val == ADMIN_CODE)

with st.expander("À propos / mode d'emploi", expanded=False):
    st.write(
        "Posez une question. Ce chatbot est configuré avec **ma logique** et **mon système de pensée** "
        "(pensée systémique, patterns, pleine conscience) et **fera le lien vers mes articles** dans ses réponses."
    )
    if CORPUS_MODE == "md":
        st.caption("📚 Corpus : lecture automatique depuis `data/corpus/*.md`.")
    elif CORPUS_MODE == "files":
        st.caption("📚 Corpus : lecture automatique depuis `data/docs/` (.odt/.pdf/.txt/.md/.html).")
    else:
        st.info("ℹ️ Aucun corpus détecté — mini-corpus de démonstration utilisé.")
    st.caption("Astuce : ajoutez `?admin=pam2025` à l’URL pour la vue admin (modifiable dans les Secrets).")

email    = st.text_input("Votre e-mail (facultatif)", placeholder="prenom.nom@email.com")
question = st.text_area("Posez votre question", placeholder="Ex : Comment appliquer les 10 piliers à mon équipe ?")
k_ctx    = st.slider("Nombre de passages (contexte)", 1, 6, 4)
go       = st.button("Rechercher", type="primary")

# Construire/charger l'index vectoriel
signature = files_signature()
M, n_docs = build_or_load_embeddings(signature)
if n_docs == 0:
    st.warning("Aucun passage indexé. Ajoute des fichiers dans `data/corpus/` ou `data/docs/` puis redémarre.")
else:
    st.caption(f"✅ Index prêt : {M.shape[0]} passages (mode: {CORPUS_MODE}).")

if go:
    if not question.strip():
        st.warning("Écris d'abord une question 🙂")
        st.stop()
    with st.spinner("Je réfléchis…"):
        ctx_docs = top_k_docs(question, k_ctx, M)
        used = []
        blocks = []
        for d in ctx_docs:
            title = d.get("title") or d["source"]
            url   = d.get("url", "")
            head  = f"{title}" + (f" <{url}>" if url else "")
            blocks.append(f"[{head}]\n\n{d['text']}")
            used.append((title, url))
        prompt_user = (
            "Contexte (extraits avec titres/liens):\n\n" +
            "\n\n---\n\n".join(blocks) +
            "\n\nQuestion:\n" + question +
            "\n\nDonne une réponse structurée, concrète, et propose des mini-exercices si utile."
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":prompt_user}],
                temperature=0.4,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()

    st.markdown("### Réponse")
    st.write(answer)

    if used:
        st.markdown("#### Sources")
        seen = set()
        for title, url in used:
            key = (title, url)
            if key in seen: 
                continue
            seen.add(key)
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")

    try:
        log_chat(email, question, answer)
        st.caption("💾 Échange enregistré (local) dans data/chat_logs.csv")
    except Exception:
        st.caption("ℹ️ Échange non journalisé (écriture non permise).")

# ================ Admin ================
if is_admin:
    st.markdown("---")
    st.subheader("👩‍💻 Admin")
    st.caption(f"Mode corpus: {CORPUS_MODE} • Docs: {len(DOCS)}")
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            st.download_button("Télécharger les logs (CSV)", f, file_name="chat_logs.csv")
    else:
        st.info("Pas encore de logs.")
