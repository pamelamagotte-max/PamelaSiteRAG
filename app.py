# app.py
import os
import time
import pathlib
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI

# --------- Réglages de base ----------
st.set_page_config(page_title="Assistant Pamela", page_icon="💬", layout="centered")

SYSTEM_PROMPT = """
Tu es l’assistant de Pamela Magotte. Tu t’exprimes simplement et avec bienveillance.
Tu relies les questions aux thèmes : pensée systémique, patterns (motifs récurrents), pleine conscience.
Tu proposes des pistes concrètes, structurées, applicables au management.
Quand c’est pertinent, tu poses 1 question courte pour clarifier avant de répondre longuement.
"""

MODEL = "gpt-5-mini"  # tu peux mettre "gpt-5" si ton quota le permet

# --------- API OpenAI ----------
# Clé lue depuis les "secrets" Streamlit Cloud (Settings > Secrets) -> OPENAI_API_KEY="sk-proj-..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("La clé OPENAI_API_KEY n'est pas configurée (Settings > Secrets).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# --------- Dossier des logs ----------
LOG_DIR = pathlib.Path("data")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "chat_logs.csv"

def append_log(email, question, answer):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "email": email.strip(),
        "question": question.strip(),
        "answer": answer.strip(),
    }
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, index=False)

# --------- UI ----------
st.markdown("### 💬 Assistant Pamela (démo en ligne)")
email = st.text_input("Votre e-mail (facultatif)", placeholder="prenom.nom@email.com")

question = st.text_input(
    "Posez votre question :",
    placeholder="Ex : Comment les 10 patterns s’appliquent au management ?"
)

ctx = st.slider("Nombre de passages (contexte)", min_value=1, max_value=5, value=3)

if st.button("Rechercher"):

    if not question.strip():
        st.warning("Écris d’abord une question 🙂")
        st.stop()

    with st.spinner("Je réfléchis…"):
        try:
            # historique minimal (pas de RAG ici, simple chat)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.6,
            )
            answer = resp.choices[0].message.content

            st.markdown("#### Réponse")
            st.write(answer)

            append_log(email, question, answer)

            st.caption("Les échanges sont enregistrés dans `data/chat_logs.csv`.")

        except Exception as e:
            st.error(f"Oups : {e}")

