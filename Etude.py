#Projet Python pour data sciences
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sb, sklearn as sk
df=pd.read_csv("Urban Air Quality and Health Impact Dataset.csv")
df.head()


#######################################2.Nettoyage#################################################################

nombre_de_lignes = df.shape[0]
print("Nombre de lignes :", nombre_de_lignes)

# Identification des valeurs manquantes
valeurs_vides = df.isnull().sum()

# On ne veut afficher que les colonnes ayant des valeurs manquantes
colonnes_vides = valeurs_vides[valeurs_vides != 0]
print(colonnes_vides)



# A présent, on veut gérer ces valeurs manquantes pour qu'elles ne faussent pas nos résultats
# Nous analysons tout d'abord la colonne 'preciptype' :

# Afficher les infos de la colonne 'preciptype'
print(df['preciptype'].describe(include='all'))

# Afficher la distribution des valeurs dans 'preciptype', y compris les valeurs manquantes
print("")
print(df['preciptype'].value_counts(dropna=False))

# Utilise la valeur la + fréquente pour remplacer les valeurs manquantes
df['preciptype'].fillna(df['preciptype'].mode()[0], inplace=True)

# Résultat
print("")
print("Résultat final :")
print(df['preciptype'].isnull().sum())



#######################################3.Analyse#################################################################

# Comptage du nombre d'occurrences de chaque ville
city_counts = df['City'].value_counts()

# Création du camembert pour la répartition des villes
plt.figure(figsize=(10, 5))

# Premier graphique : répartition des villes
plt.subplot(1, 2, 1)
plt.pie(city_counts, labels=city_counts.index, autopct='%1.1f%%', startangle=140, colors=sb.color_palette("pastel"))
plt.title("Répartition des villes")

# Deuxième graphique : répartition des sources
source_counts = df['source'].value_counts()
plt.subplot(1, 2, 2)
plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=140, colors=sb.color_palette("muted"))
plt.title("Répartition des sources")

plt.tight_layout()
plt.show()