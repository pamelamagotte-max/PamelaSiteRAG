# app.py
import os
import time
import csv
from datetime import datetime
import numpy as np
import streamlit as st
from openai import OpenAI

# ---------- Secrets & config ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
ADMIN_CODE = st.secrets.get("ADMIN_CODE", "pam2025")

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY n'est pas configurée dans les Secrets Streamlit.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Dossier de logs
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "chat_logs.csv")

# ---------- Identité & style ----------
SYSTEM_PROMPT = (
    "Tu es l'assistant de Pamela Magotte. Ta voix est claire, bienveillante et structurée. "
    "Tu relies les sujets à la pensée systémique, aux patterns (motifs récurrents) et à la pleine conscience. "
    "Tu simplifies sans appauvrir, proposes des exemples concrets et des mini-exercices. "
    "Si une question sort de ton périmètre, tu le dis et tu proposes une reformulation utile."
)

# ---------- Mini corpus ----------
CORPUS = [
    {
        "source": "10_piliers_intro",
        "text": (
            "Les 10 piliers proposent une grille systémique pour relier perception, action et responsabilité. "
            "Ils servent à voir les patterns (récurrences) dans les comportements et organisations."
        ),
    },
    {
        "source": "pleine_conscience_base",
        "text": (
            "La pleine conscience est une capacité d'attention ouverte et non-jugeante. "
            "Elle aide à voir les boucles automatiques et à choisir une réponse plutôt qu'une réaction."
        ),
    },
    {
        "source": "management_pattern",
        "text": (
            "En management, repérer les patterns permet d'intervenir au bon niveau du système : "
            "structure, règles du jeu, langage, rituels, feedbacks et apprentissage."
        ),
    },
]

# Map des sources -> URLs de tes articles (remplace par les liens réels)
SOURCE_LINKS = {
    "10_piliers_intro": "https://ton-site/article-10-piliers",
    "pleine_conscience_base": "https://ton-site/article-pleine-conscience",
    "management_pattern": "https://ton-site/article-management-patterns",
}

# ---------- Embeddings & recherche ----------
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    # Transforme une liste de textes en vecteurs
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in resp.data], dtype=float)

@st.cache_data(show_spinner=False)
def corpus_matrix():
    texts = [doc["text"] for doc in CORPUS]
    return embed_texts(texts)

def embed_query(q):
    r = client.embeddings.create(model="text-embedding-3-small", input=[q])
    return np.array(r.data[0].embedding, dtype=float)

def top_k_context(query, k=3):
    """Retourne (contexte_concaténé, liste_des_sources_utilisées)"""
    M = corpus_matrix()
    qv = embed_query(query)
    # cosine similarity
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-9)
    idxs = np.argsort(-sims)[:k]

    ctx_parts, used_sources = [], []
    for i in idxs:
        src = CORPUS[i]['source']
        used_sources.append(src)
        ctx_parts.append(f"[{src}] {CORPUS[i]['text']}")

    return "\n\n".join(ctx_parts), used_sources

# ---------- Log ----------
def log_chat(email, question, answer):
    newfile = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["timestamp", "email", "question", "answer"])
        w.writerow([datetime.utcnow().isoformat(), email or "", question, answer])

# ---------- UI ----------
st.set_page_config(page_title="Assistant Pamela", page_icon="💬", layout="centered")
st.title("💬 Assistant Pamela")

# Param admin dans l'URL : ?admin=pam2025
# (Nouvelle API Streamlit : st.query_params)
admin_val = st.query_params.get("admin", "")
is_admin = (admin_val == ADMIN_CODE)

with st.expander("À propos / mode d'emploi", expanded=False):
    st.write(
        "Posez une question. Ce chatbot est configuré avec **ma logique** et **mon système de pensée** "
        "(pensée systémique, patterns, pleine conscience) et **fera le lien vers mes articles** dans ses réponses."
    )
    st.caption("Astuce : ajoutez `?admin=pam2025` à l’URL pour la vue admin (modifiable dans les Secrets).")

email = st.text_input("Votre e-mail (facultatif)", placeholder="prenom.nom@email.com")
question = st.text_area("Posez votre question", placeholder="Ex : Comment appliquer les 10 piliers à mon équipe ?")
k_ctx = st.slider("Nombre de passages (contexte)", 1, 5, 3)

go = st.button("Rechercher", type="primary")

if go:
    if not question.strip():
        st.warning("Écris d'abord une question 🙂")
        st.stop()

    with st.spinner("Je réfléchis…"):
        # Récup du contexte + liste des sources utilisées
        context, used_sources = top_k_context(question, k=k_ctx)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Contexte (sources):\n"
                f"{context}\n\n"
                f"Question:\n{question}\n\n"
                "Réponds de façon structurée, avec exemples et mini-exercices si utiles."
            )},
        ]

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.4,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()

    st.markdown("### Réponse")
    st.write(answer)

    # 🔗 Affichage des sources citées avec liens (si dispo)
    if used_sources:
        st.markdown("#### 🔗 Sources citées / Aller plus loin")
        # On garde l'ordre et enlève les doublons
        seen = set()
        ordered_unique = []
        for s in used_sources:
            if s not in seen:
                seen.add(s)
                ordered_unique.append(s)

        for src in ordered_unique:
            url = SOURCE_LINKS.get(src, "")
            if url:
                st.markdown(f"- **{src}** → [{url}]({url})")
            else:
                st.markdown(f"- **{src}**")

    # Log (tolérant si écriture interdite)
    try:
        log_chat(email, question, answer)
        st.caption("✅ Échange enregistré (local) dans data/chat_logs.csv")
    except Exception:
        st.caption("ℹ️ Échange non journalisé (écriture non permise).")

# ---------- Vue admin ----------
if is_admin:
    st.markdown("---")
    st.subheader("👩‍💻 Admin")
    st.caption("Derniers échanges")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            st.download_button("Télécharger les logs (CSV)", f, file_name="chat_logs.csv")
    else:
        st.info("Pas encore de logs.")


