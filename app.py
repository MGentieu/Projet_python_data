import pandas as pd
import streamlit as st
import numpy as np
import pickle

#pip install streamlit
"""import os
os.system("streamlit run app_streamlit.py")
"""
"""from pyngrok import ngrok
import subprocess
# Démarrer Streamlit en arrière-plan
process = subprocess.Popen(["streamlit", "run", "app_streamlit.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Lancer un tunnel avec ngrok
public_url = ngrok.connect(port=8501)
print(f"Streamlit app running at: {public_url}")
"""
#process.terminate()


# Charger le modèle sauvegardé
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
# Charger le préprocesseur (si nécessaire, sauvegardez-le comme le modèle)
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Titre de l'application
st.title("Prédiction du prix de la maison")

# Description de l'application
st.write("""
Cette application utilise un modèle de régression pour prédire le prix d'une maison 
en fonction de ses caractéristiques.
""")

# Entrées utilisateur
st.sidebar.header("Entrées utilisateur")
nouvelle_maison = {
    "bedrooms": st.sidebar.number_input("Nombre de chambres", min_value=1, step=1),
    "bathrooms": st.sidebar.number_input("Nombre de salles de bain", min_value=1, step=1),
    "sqft_living": st.sidebar.number_input("Superficie habitable (en sqft)", min_value=1, step=1),
    "sqft_lot": st.sidebar.number_input("Taille du terrain (en sqft)", min_value=1, step=1),
    "floors": st.sidebar.number_input("Nombre d'étages", min_value=1, step=1),
    "waterfront": st.sidebar.selectbox("Bord de l'eau (1: Oui, 0: Non)", [0, 1]),
    "view": st.sidebar.number_input("Vue (0 à 4)", min_value=0, max_value=4, step=1),
    "condition": st.sidebar.number_input("Condition (1 à 5)", min_value=1, max_value=5, step=1),
    "grade": st.sidebar.number_input("Qualité (1 à 13)", min_value=1, max_value=13, step=1),
    "sqft_above": st.sidebar.number_input("Superficie hors sous-sol (en sqft)", min_value=1, step=1),
    "sqft_basement": st.sidebar.number_input("Superficie du sous-sol (en sqft)", min_value=0, step=1),
    "yr_built": st.sidebar.number_input("Année de construction", min_value=1800, max_value=2024, step=1),
    "yr_renovated": st.sidebar.number_input("Année de rénovation (0 si aucune)", min_value=0, max_value=2024, step=1),
    "zipcode": st.sidebar.text_input("Code postal", value="98178"),
    "lat": st.sidebar.number_input("Latitude", value=47.5112, step=0.0001),
    "long": st.sidebar.number_input("Longitude", value=-122.257, step=0.0001),
    "sqft_living15": st.sidebar.number_input("Superficie habitable moyenne des 15 maisons voisines", min_value=1, step=1),
    "sqft_lot15": st.sidebar.number_input("Taille moyenne du terrain des 15 maisons voisines", min_value=1, step=1)
}

# Bouton de prédiction
if st.sidebar.button("Prédire le prix"):
    # Convertir les entrées utilisateur en DataFrame
    nouvelle_maison_df = pd.DataFrame([nouvelle_maison])

    # Transformer les données utilisateur avec le préprocesseur
    nouvelle_maison_encoded = preprocessor.transform(nouvelle_maison_df)

    # Prédire le prix
    prix_estime = model.predict(nouvelle_maison_encoded)[0]

    # Afficher le résultat
    st.success(f"Le prix estimé de la maison est : ${prix_estime:,.2f}")

# Ajouter une section pour l'explication du modèle
st.write("### À propos du modèle")
st.write("""
Le modèle utilisé ici est un modèle de régression linéaire. Vous pouvez ajuster les
paramètres en fonction des caractéristiques de votre propre modèle.
""")
