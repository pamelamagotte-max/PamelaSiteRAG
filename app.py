import streamlit as st
from openai import OpenAI

# Récupération de la clé API depuis secrets (configurée dans Streamlit Cloud)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Test OpenAI avec Streamlit 🚀")

prompt = st.text_input("Pose une question à l'IA :", "Bonjour, qui es-tu ?")

if st.button("Envoyer"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write("### Réponse :")
    st.write(response.choices[0].message.content)
