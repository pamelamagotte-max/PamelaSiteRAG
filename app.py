# app.py — Assistant Pamela (robuste, anti-RateLimit, lecture auto du corpus)
import os
import re
import csv
import glob
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from openai import OpenAI

# =========================
# Secrets & config
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
ADMIN_CODE     = st.secrets.get("ADMIN_CODE", "pam2025")

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY n'est pas configurée dans les Secrets Streamlit.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

DATA_DIR   = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
EMB_FILE   = DATA_DIR / "embeddings.npy"
EMB_META   = DATA_DIR / "embeddings_meta.json"
LOG_PATH   = DATA_DIR / "chat_logs.csv"

DATA_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Identité & style
# =========================
SYSTEM_PROMPT = (
    "Tu es l'assistant de Pamela Magotte. Voix claire, bienveillante et structurée. "
    "Tu relies les sujets à la pensée systémique, aux patterns (motifs récurrents) et à la pleine conscience. "
    "Tu simplifies sans appauvrir, donnes des exemples concrets et des mini-exercices. "
    "Si une question sort de ton périmètre, tu le dis et proposes une reformulation utile."
)

# =========================
# Fallback mini-corpus (si aucun .md)
# =========================
FALLBACK = [
    {
        "source": "10_piliers_intro",
        "title":  "Les 10 piliers – introduction",
        "url":    "",
        "text": (
            "Les 10 piliers proposent une grille systémique pour relier perception, action et responsabilité. "
            "Ils servent à voir les patterns (récurrences) dans les comportements et organisations."
        ),
    },
    {
        "source": "pleine_conscience_base",
        "title":  "Pleine conscience – bases",
        "url":    "",
        "text": (
            "La pleine conscience est une capacité d'attention ouverte et non-jugeante. "
            "Elle aide à voir les boucles automatiques et à choisir une réponse plutôt qu'une réaction."
        ),
    },
    {
        "source": "management_pattern",
        "title":  "Management & patterns",
        "url":    "",
        "text": (
            "En management, repérer les patterns permet d'intervenir au bon niveau du système : "
            "structure, règles du jeu, langage, rituels, feedbacks et apprentissage."
        ),
    },
]

# =========================
# Lecture du corpus .md
# =========================
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

def _parse_frontmatter(raw: str):
    """
    Lit un front-matter minimal:
    ---
    title: "Mon titre"
    url: "https://..."
    ---
    Corps...
    """
    meta = {}
    body = raw
    m = FRONTMATTER_RE.match(raw)
    if m:
        fm = m.group(1)
        body = raw[m.end():]
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body.strip()

def load_corpus_from_md(max_files: int = 1000, max_chars_per_doc: int = 8000):
    """
    Charge tous les .md de data/corpus/.
    Retourne (docs, has_md) où docs = [{source,title,url,text}, ...]
    """
    paths = sorted(set(glob.glob(str(CORPUS_DIR / "*.md"))))
    docs = []
    for p in paths[:max_files]:
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title") or Path(p).stem
        url   = meta.get("url", "")
        # tronque par sécurité (on découpera côté modèle au besoin)
        text  = body[:max_chars_per_doc].strip()
        if len(text) < 200:  # ignore fichiers trop courts/bruit
            continue
        docs.append({
            "source": Path(p).name,  # identifiant = nom du fichier
            "title":  title,
            "url":    url,
            "text":   text,
        })
    if docs:
        return docs, True
    # sinon fallback
    return FALLBACK, False

DOCS, HAS_MD = load_corpus_from_md()

# =========================
# Embeddings robustes (lots + retry + cache disque)
# =========================
EMB_MODEL = "text-embedding-3-small"

def _texts_signature(texts: list[str]) -> str:
    # Empreinte du corpus => si ça ne change pas, on relit du disque
    h = hashlib.md5()
    h.update(str(len(texts)).encode())
    for t in texts[:200]:   # limite pour rester léger
        h.update(str(len(t)).encode())
        h.update(b"|")
    sample = "\n".join(t[:300] for t in texts[:30])
    h.update(sample.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def _backoff_sleep(attempt: int):
    # 0.5 -> 1 -> 2 -> 4 -> 8 -> 8 s
    time.sleep(min(8, 0.5 * (2 ** attempt)))

def _embed_in_batches(texts: list[str], batch: int = 64) -> np.ndarray:
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
                _backoff_sleep(attempt)
        time.sleep(0.2)  # politesse entre lots
    return np.array(vecs, dtype=float)

@st.cache_data(show_spinner=True)
def build_or_load_embeddings(signature: str):
    """
    Calcule ou relit les embeddings:
    - si EMB_FILE + EMB_META avec signature identique -> charge numpy
    - sinon -> calcule en lots + écrit sur disque
    Renvoie: (M, n_texts) avec M shape (N, D)
    """
    if EMB_FILE.exists() and EMB_META.exists():
        try:
            meta = json.loads(EMB_META.read_text(encoding="utf-8"))
            if meta.get("sig") == signature and meta.get("n") == len(DOCS):
                M = np.load(EMB_FILE)
                return M, len(DOCS)
        except Exception:
            pass

    texts = [d["text"] for d in DOCS]
    # sécurité: si très gros corpus, tu peux limiter ici puis augmenter ensuite
    # ex: texts = texts[:500]
    M = _embed_in_batches(texts, batch=64)
    try:
        np.save(EMB_FILE, M)
        EMB_META.write_text(json.dumps({"sig": signature, "n": len(DOCS)}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # hébergeur en read-only → on n'arrête pas l'app pour ça
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
            _backoff_sleep(attempt)

def top_k_docs(query: str, k: int, M: np.ndarray):
    if M.shape[0] == 0:
        return []
    qv = embed_query(query)
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-9)
    idx = np.argsort(-sims)[:k]
    return [DOCS[i] for i in idx]

# =========================
# Log
# =========================
def log_chat(email, question, answer):
    newfile = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["timestamp", "email", "question", "answer"])
        w.writerow([datetime.utcnow().isoformat(), email or "", question, answer])

# =========================
# UI
# =========================
st.set_page_config(page_title="Assistant Pamela", page_icon="💬", layout="centered")
st.title("💬 Assistant Pamela")

# Param admin dans l'URL : ?admin=pam2025
admin_val = st.query_params.get("admin", "")
is_admin = (admin_val == ADMIN_CODE)

with st.expander("À propos / mode d'emploi", expanded=False):
    st.write(
        "Posez une question. Ce chatbot est configuré avec **ma logique** et **mon système de pensée** "
        "(pensée systémique, patterns, pleine conscience) et **fera le lien vers mes articles** dans ses réponses."
    )
    if HAS_MD:
        st.caption("📚 Corpus : lecture automatique depuis `data/corpus/*.md`.")
    else:
        st.info("ℹ️ Aucun `.md` détecté dans `data/corpus/` — mini-corpus de démonstration utilisé.")
    st.caption("Astuce : ajoutez `?admin=pam2025` à l’URL pour la vue admin (modifiable dans les Secrets).")

email    = st.text_input("Votre e-mail (facultatif)", placeholder="prenom.nom@email.com")
question = st.text_area("Posez votre question", placeholder="Ex : Comment appliquer les 10 piliers à mon équipe ?")
k_ctx    = st.slider("Nombre de passages (contexte)", 1, 6, 4)

go = st.button("Rechercher", type="primary")

# Construit / charge les embeddings (avec cache disque)
sig = hashlib.md5(("|".join(sorted(d["source"] for d in DOCS)) + f":{len(DOCS)}").encode()).hexdigest()
M, n_texts = build_or_load_embeddings(sig)
if n_texts == 0:
    st.warning("Aucun passage indexé. Ajoute des fichiers `.md` dans `data/corpus/`.")
else:
    st.caption(f"✅ Index prêt : {M.shape[0]} passages.")

if go:
    if not question.strip():
        st.warning("Écris d'abord une question 🙂")
        st.stop()

    with st.spinner("Je réfléchis…"):
        ctx_docs = top_k_docs(question, k_ctx, M)
        # Construit le contexte + mémorise les sources
        used = []
        context_blocks = []
        for d in ctx_docs:
            title = d.get("title") or d["source"]
            url   = d.get("url", "")
            head  = f"{title}"
            if url:
                head += f" <{url}>"
            context_blocks.append(f"[{head}]\n\n{d['text']}")
            used.append((title, url))

        user_prompt = (
            "Contexte (extraits avec titres/liens):\n\n" +
            "\n\n---\n\n".join(context_blocks) +
            "\n\nQuestion:\n" + question +
            "\n\nDonne une réponse structurée, concrète, et propose des mini-exercices si utile."
        )

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
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
        for (title, url) in used:
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")

    # Log
    try:
        log_chat(email, question, answer)
        st.caption("💾 Échange enregistré (local) dans data/chat_logs.csv")
    except Exception:
        st.caption("ℹ️ Échange non journalisé (écriture non permise).")

# =========================
# Vue admin
# =========================
if is_admin:
    st.markdown("---")
    st.subheader("👩‍💻 Admin")
    st.caption(f"Docs chargés: {len(DOCS)} • .md détectés: {HAS_MD}")
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            st.download_button("Télécharger les logs (CSV)", f, file_name="chat_logs.csv")
    else:
        st.info("Pas encore de logs.")

