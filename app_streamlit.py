import pandas as pd
import streamlit as st
import pickle

#pip install streamlit
"""import os
os.system("streamlit run app_streamlit.py")
"""
"""
from pyngrok import ngrok
import subprocess
import time

# Démarrer Streamlit en arrière-plan
process = subprocess.Popen(["streamlit", "run", "app_streamlit.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Lancer un tunnel avec ngrok
public_url = ngrok.connect(port=8501)
print(f"Streamlit app running at: {public_url}")

time.sleep(10)

process.terminate()
"""


# Charger le modèle sauvegardé et le préprocesseur
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

st.title("Prédiction du prix de la maison")

st.write("""
Cette application utilise un modèle de régression pour prédire le prix d'une maison 
en fonction de ses caractéristiques.
""")

st.sidebar.header("Entrées utilisateur")
nouvelle_maison = {
    "bathrooms": st.sidebar.number_input("Nombre de salles de bain", min_value=1, step=1),
    "sqft_living": st.sidebar.number_input("Superficie habitable (en sqft)", min_value=1, step=1),
    "grade": st.sidebar.number_input("Qualité (1 à 13)", min_value=1, max_value=13, step=1),
    "sqft_above": st.sidebar.number_input("Superficie hors sous-sol (en sqft)", min_value=1, step=1),
    "lat": st.sidebar.number_input("Latitude", value=47.5112, step=0.0001),
    "sqft_living15": st.sidebar.number_input("Superficie habitable moyenne des 15 maisons voisines", min_value=1, step=1),
}

# Bouton de prédiction
if st.sidebar.button("Prédire le prix"):
    nouvelle_maison_df = pd.DataFrame([nouvelle_maison])
    nouvelle_maison_encoded = preprocessor.transform(nouvelle_maison_df)
    prix_estime = model.predict(nouvelle_maison_encoded)[0]
    st.success(f"Le prix estimé de la maison est : ${prix_estime:,.2f}")

st.write("### À propos du modèle")
st.write("""
Le modèle utilisé ici est un modèle de régression linéaire. Vous pouvez ajuster les
paramètres en fonction des caractéristiques de votre propre modèle.
""")
