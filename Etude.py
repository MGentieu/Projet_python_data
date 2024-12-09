# Projet Python pour data sciences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df=pd.read_csv("Engineering_graduate_salary.csv")


def affichage_graph_reussite():
    # Prédictions sur l'ensemble de test
    y_test_pred = model.predict(X_test_scaled)

    # Graphique 1: Prédictions vs Réels
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ligne d'identité
    plt.title('Prédictions vs Valeurs réelles')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.show()

    # Graphique 2: Résidus
    residuals = y_test - y_test_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Graphique des résidus')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.show()

    # Graphique 3: Histogramme des résidus
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='blue', bins=30)
    plt.title('Histogramme des résidus')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.show()

def affichage_pourcentage_reussite():
    from sklearn.metrics import mean_absolute_error

    # Calcul du R²
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Affichage des performances
    print(f"Performances sur l'ensemble d'entraînement :")
    print(f"  - RMSE : {train_rmse:.2f}")
    print(f"  - R²   : {train_r2:.2f}")
    print(f"  - Pourcentage de réussite sur l'entraînement (R²) : {train_r2 * 100:.2f}%")

    print(f"\nPerformances sur l'ensemble de test :")
    print(f"  - RMSE : {test_rmse:.2f}")
    print(f"  - R²   : {test_r2:.2f}")
    print(f"  - Pourcentage de réussite sur le test (R²) : {test_r2 * 100:.2f}%")

    # Calcul de l'Erreur Absolue Moyenne (MAE) et du pourcentage de réussite
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Affichage des MAE
    print(f"\nErreur absolue moyenne sur l'entraînement : {train_mae:.2f}")
    print(f"Erreur absolue moyenne sur le test : {test_mae:.2f}")


def predire_salaire(new_employee):
    # Créer un DataFrame avec les données saisies

    new_data_df = pd.DataFrame(new_employee)

    # Appliquer le même prétraitement que pour X_train
    new_data_encoded = preprocessor.transform(new_data_df)  # Encodage
    new_data_scaled = scaler.transform(new_data_encoded)  # Mise à l'échelle

    # Faire la prédiction
    predicted_salary = model.predict(new_data_scaled)

    # Afficher le salaire prédit
    print(f"\nLe salaire prédit pour cette personne est : {predicted_salary[0]:.2f} INR")

#######################################2. Nettoyage#################################################################
"""
# Nettoyage des données (à compléter selon vos besoins)
df = df[df['collegeGPA']>20]
#df = df[df['GraduationYear'] >= 1950]
#df['GraduationYear'] = df['GraduationYear'].astype(int)
df = df.drop(columns=['CollegeCityID'])
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['YearOfBirth'] = df['DOB'].dt.year

df['12board_category'] = df['12board'].apply(lambda x: 'CBSE' if 'cbse' in str(x)
                                              else 'state board' if 'state' in str(x)
                                              else 'Autres')
"""
mean_salary = df['Salary'].mean()
df['Salary'] = df['Salary'].apply(lambda x: mean_salary if x > 1500000 else x)



#######################################3. Analyse#################################################################
"""
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

##################  salaires supérieurs à 1 million  ##################
df_filtered = df[df['Salary'] > 1000000]

# Créer un graphique des salaires par genre avec le nombre de personnes
plt.figure(figsize=(8, 6))

# Tracer un countplot avec le salaire sur l'axe des X et le nombre de personnes sur l'axe des Y
sb.countplot(x='Salary', hue='Gender', data=df_filtered, palette='Set2', dodge=True)
plt.xticks(rotation=45)  # Incliner les labels des abscisses de 45 degrés

# Paramètres du graphique
plt.title("Nombre de personnes avec un salaire supérieur à 1 million par genre")
plt.xlabel("Salaire")
plt.ylabel("Nombre de personnes")

# Affichage du graphique
plt.tight_layout()
plt.show()"""


#######################################4. Analyse#################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Supposons que vous souhaitiez prédire le salaire avec les autres colonnes comme caractéristiques
X = df.drop(columns=['Salary', 'DOB', 'openess_to_experience', 'nueroticism', 'extraversion', 'agreeableness', 'conscientiousness', 'CollegeID', '10board', '12board', 'CollegeState'])  # Caractéristiques (features), en excluant la colonne cible
#X = df.loc[:, ['Degree', 'Specialization', '12percentage']]
y = df['Salary']  # Variable cible (target)

y_bins = pd.qcut(y, q=4, labels=False)
# Diviser les données en ensemble d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_bins)


print("Ensemble d'entraînement :", X_train.shape)
print("Ensemble de test :", X_test.shape)


# Identifier les colonnes catégoriques
categorical_columns = X.select_dtypes(include=['object']).columns
# OneHotEncoding pour les colonnes catégoriques
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)], remainder='passthrough')

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


# Standardiser les données (calculer la moyenne et l'écart-type pour chaque colonne, puis transformer les données)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


#######################################5. Entraienemnt#################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train_scaled, y_train)

# Prédire les valeurs sur l'ensemble d'entraînement et de test
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Évaluer les performances du modèle
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)  # RMSE sur train
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)  # RMSE sur test
train_r2 = r2_score(y_train, y_train_pred)  # R² sur train
test_r2 = r2_score(y_test, y_test_pred)  # R² sur test

# Afficher les métriques de performance
print(f"Performances sur l'ensemble d'entraînement :")
print(f"  - RMSE : {train_rmse:.2f}")
print(f"  - R²   : {train_r2:.2f}")

print(f"\nPerformances sur l'ensemble de test :")
print(f"  - RMSE : {test_rmse:.2f}")
print(f"  - R²   : {test_r2:.2f}")


affichage_graph_reussite()
#affichage_pourcentage_reussite()






new_employee = {
        "ID": [604399],
        "Gender": ["f"],
        "10percentage": [87.8],
        "12graduation": [2009],
        "12percentage": [84.0],
        "CollegeTier": [1],
        "Degree": ["B.Tech/B.E."],
        "Specialization": ["instrumentation and control engineering"],
        "collegeGPA": [73.82],
        "CollegeCityID": [6920],
        "CollegeCityTier": [1],
        "GraduationYear": [2013],
        "English": [650],
        "Logical": [665],
        "Quant": [810],
        "Domain": [0.694479327708463],
        "ComputerProgramming": [485],
        "ElectronicsAndSemicon": [366],
        "ComputerScience": [-1],
        "MechanicalEngg": [-1],
        "ElectricalEngg": [-1],
        "TelecomEngg": [-1],
        "CivilEngg": [-1]
    }
# Appeler la fonction pour demander les informations et faire la prédiction
predire_salaire(new_employee)
