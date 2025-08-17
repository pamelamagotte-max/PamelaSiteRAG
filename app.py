import streamlit as st
from openai import OpenAI

# Charger la clé depuis secrets.toml
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Titre de ton app
st.title("🤖 Mon IA perso")

# Zone pour écrire la question
user_input = st.text_input("Pose-moi une question :")

# Quand l’utilisateur envoie du texte
if st.button("Envoyer") and user_input:
    try:
        # Appel à l'API OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # rapide et économique
            messages=[
                {"role": "system", "content": "Tu es une IA simple et sympa intégrée sur mon site."},
                {"role": "user", "content": user_input}
            ]
        )

        # Affichage de la réponse
        st.success(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Erreur : {e}")
