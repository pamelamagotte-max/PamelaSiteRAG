# 03_app_streamlit.py — Version ultra-légère (pas d'install lourde)
import streamlit as st
import json, os, datetime, csv

# Code admin pour voir l'historique : ajoute ?admin=pam2025 à l'URL
ADMIN_CODE = os.getenv("ADMIN_CODE", "pam2025")

# Mini "corpus" de démo (remplaçable plus tard par ton site)
CORPUS = [
    {
        "id": "patterns",
        "url": "https://www.pamelamagotte.fr/les-10-piliers-fondements-de-mon-approche/",
        "text": "Les 10 patterns universels du vivant permettent de lire la complexité sans réductionnisme. Equilibres dynamiques, boucles de rétroaction, émergence."
    },
    {
        "id": "pensee_systemique",
        "url": "https://www.pamelamagotte.fr/la-pensee-systemique/",
        "text": "La pensée systémique relie les éléments, observe les liens, les boucles, la régulation, et les propriétés émergentes des systèmes vivants."
    },
    {
        "id": "science_conscience",
        "url": "https://www.pamelamagotte.fr/science-et-conscience/",
        "text": "Relier science et conscience avec rigueur vivante : cohérence, alignement, responsabilité individuelle et collective."
    }
]

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def naive_retrieve(query, k=3):
    # Petit score naïf par mots en commun (pour démo)
    q = [w.lower() for w in query.split() if len(w) > 2]
    scores = []
    for item in CORPUS:
        t = (item["text"] + " " + item["id"]).lower()
        s = sum(t.count(w) for w in q)
        scores.append((s, item))
    scores.sort(key=lambda x: x[0], reverse=True)
    hits = [it for sc,it in scores if sc > 0][:k]
    if not hits:
        hits = [c for c in CORPUS][:min(k, len(CORPUS))]
    return hits

def synthesize(question, hits):
    # Réponse style "Pamela" (démo), avec sources
    points = "\n".join([f"- {h['text']}" for h in hits])
    sources = "\n".join([f"- {h['url']}" for h in hits])
    return f"""**Synthèse (démo locale)**  
Question : {question}

Lecture systémique :
{points}

Approche :
- Clarifier le cadre (patterns, boucles de rétroaction, émergences).
- Proposer une action simple alignée (cohérence, responsabilité, pleine conscience).

Sources :
{sources}"""

def log_interaction(email, query, answer, hits):
    ensure_data_dir()
    log = {
        "time": datetime.datetime.utcnow().isoformat(),
        "email": email or "",
        "query": query,
        "answer": answer,
        "sources": [h["url"] for h in hits]
    }
    # JSONL
    with open("data/chat_logs.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    # CSV
    csv_exists = os.path.exists("data/chat_logs.csv")
    with open("data/chat_logs.csv","a",encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not csv_exists:
            w.writerow(["time","email","query","answer","sources"])
        w.writerow([log["time"], log["email"], log["query"], log["answer"], " | ".join(log["sources"])])

def read_logs():
    ensure_data_dir()
    p = "data/chat_logs.jsonl"
    if not os.path.exists(p): return []
    rows = []
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    return rows

# ---------------- UI ----------------
st.set_page_config(page_title="Assistant Pamela (démo)", layout="wide")
st.title("💬 Assistant Pamela (démo locale)")

# Admin caché
try:
    qp = st.query_params
    admin_param = qp.get("admin", None)
except Exception:
    qp = st.experimental_get_query_params()
    admin_param = qp.get("admin",[None])[0] if "admin" in qp else None

if admin_param and admin_param == ADMIN_CODE:
    st.subheader("Historique des échanges (admin)")
    data = read_logs()
    if data:
        st.dataframe(data, use_container_width=True)
        if os.path.exists("data/chat_logs.csv"):
            with open("data/chat_logs.csv","rb") as f:
                st.download_button("Télécharger le CSV", data=f, file_name="chat_logs.csv", mime="text/csv")
    else:
        st.info("Aucun échange enregistré pour le moment.")
    st.stop()

email = st.text_input("Votre e-mail (facultatif)", value="", placeholder="prenom.nom@email.com")
q = st.text_input("Posez votre question :", placeholder="Ex : Comment les 10 patterns s’appliquent au management ?")
k = st.slider("Nombre de passages (contexte)", 1, 5, 3)

if st.button("Rechercher") and q.strip():
    hits = naive_retrieve(q, k=k)
    with st.expander("Passages retrouvés (démo)"):
        for h in hits:
            st.markdown(f"- [{h['id']}]({h['url']}) — {h['text']}")
    answer = synthesize(q, hits)
    st.markdown("### Réponse")
    st.write(answer)
    log_interaction(email, q, answer, hits)

st.caption("Démo locale très légère (sans install d'IA). Les échanges sont enregistrés dans data/chat_logs.csv. Ajoute ?admin=pam2025 à l'URL pour voir l'historique.")
