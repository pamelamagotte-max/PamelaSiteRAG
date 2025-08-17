# app.py — Assistant Pamela (IA simplifiée)

import os, csv, datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Assistant Pamela", page_icon="🧠", layout="centered")

# --- OpenAI client : on lit la clé depuis Streamlit Secrets OU variable d'env ---
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Clé OpenAI manquante. Ajoute OPENAI_API_KEY dans 'Secrets' Streamlit.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Persona : tu peux l'ajuster librement ---
SYSTEM_PROMPT = """
Tu es l'assistant cognitif de Pamela Magotte.
Style : optimiste, humble, tourné vers l’avenir. Ton langage est naturel et professionnel.
Cadre : pensée systémique, 10 patterns universels du vivant, science et conscience.
Réponds de façon claire, structurée et actionnable. Si une info manque, dis-le simplement.
"""

def ask_llm(question: str) -> str:
    # Modèle : mini et économique ; tu peux changer plus tard
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
    )
    return resp.choices[0].message.content

def log_interaction(email, question, answer):
    try:
        os.makedirs("data", exist_ok=True)
        p = "data/chat_logs.csv"
        new_file = not os.path.exists(p)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["ts_utc", "email", "question", "answer"])
            w.writerow([datetime.datetime.utcnow().isoformat(), email or "", question, answer])
    except Exception:
        pass  # en prod simple : on ignore les erreurs de log

st.title("Assistant Pamela (IA)")
email = st.text_input("Votre e-mail (facultatif)")
question = st.text_area(
    "Posez votre question :",
    placeholder="Ex : Comment les 10 patterns s’appliquent au management ?",
    height=140,
)

if st.button("Répondre", type="primary"):
    if not question.strip():
        st.warning("Écris une question 😉")
        st.stop()
    with st.spinner("Je réfléchis…"):
        answer = ask_llm(question)
    st.markdown("### Réponse")
    st.write(answer)
    log_interaction(email, question, answer)

st.caption("Démonstration : les échanges sont enregistrés localement (data/chat_logs.csv).")
