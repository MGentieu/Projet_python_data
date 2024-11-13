# Projet Python pour data sciences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Chargement des données
df=pd.read_csv("Engineering_graduate_salary.csv")

#######################################2. Nettoyage#################################################################

# Nettoyage des données (à compléter selon vos besoins)

#######################################3. Analyse#################################################################

plt.figure(figsize=(8, 8))
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['orange', 'green'])
plt.title("Répartition des Genres")
plt.ylabel("")  # Enlever l'étiquette de l'axe y pour un meilleur affichage


##################  Distribution des scores (valeurs) ##################
# Création de la figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphe pour les scores du brevet (10ème année)
sb.scatterplot(data=df, x='10percentage', y='ID', hue='Gender', style='Gender', ax=axes[0])
axes[0].set_title("Distribution des scores de 10ème année (brevet)")
axes[0].set_xlabel("Score en 10ème année (%)")
axes[0].set_ylabel("Nombre de candidats")

# Graphe pour les scores du bac (12ème année)
sb.scatterplot(data=df, x='12percentage', y='ID', hue='Gender', style='Gender', ax=axes[1])
axes[1].set_title("Distribution des scores de 12ème année (bac)")
axes[1].set_xlabel("Score en 12ème année (%)")
axes[1].set_ylabel("Nombre de candidats")

# Affichage de la légende et des graphes
plt.legend(title="Sexe", loc="upper right")
plt.tight_layout()


##################  Distribution des scores (densité)  ##################
# Création de la figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Préparation des données pour les scores de 10ème année
score_10_m = df[df['Gender'] == 'm']['10percentage']
score_10_f = df[df['Gender'] == 'f']['10percentage']

# Créer des lignes de densité pour les scores de 10ème année
sb.kdeplot(score_10_m, label='Hommes', ax=axes[0], color='blue')
sb.kdeplot(score_10_f, label='Femmes', ax=axes[0], color='pink')
axes[0].set_title("Distribution des scores en 10ème année (brevet)")
axes[0].set_xlabel("Score en 10ème année (%)")
axes[0].set_ylabel("Densité")
axes[0].legend()

# Préparation des données pour les scores de 12ème année
score_12_m = df[df['Gender'] == 'm']['12percentage']
score_12_f = df[df['Gender'] == 'f']['12percentage']

# Créer des lignes de densité pour les scores de 12ème année
sb.kdeplot(score_12_m, label='Hommes', ax=axes[1], color='blue')
sb.kdeplot(score_12_f, label='Femmes', ax=axes[1], color='pink')
axes[1].set_title("Distribution des scores en 12ème année (bac)")
axes[1].set_xlabel("Score en 12ème année (%)")
axes[1].set_ylabel("Densité")
axes[1].legend()

plt.tight_layout()

##################  Distribution des salaires (densité)  ##################

# Création de la figure avec un sous-graphe pour les salaires
plt.figure(figsize=(8, 6))

# Création des courbes de densité pour les salaires
salary_m = df[df['Gender'] == 'm']['Salary']
salary_f = df[df['Gender'] == 'f']['Salary']

# Tracer les lignes de densité pour les salaires
sb.kdeplot(salary_m, label='Hommes', color='blue')
sb.kdeplot(salary_f, label='Femmes', color='pink')

# Paramètres du graphique
plt.title("Distribution des salaires par sexe")
plt.xlabel("Salaire")
plt.ylabel("Densité")
plt.legend()
plt.tight_layout()
plt.show()