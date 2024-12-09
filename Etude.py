import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df=pd.read_csv("kc_house_data.csv")


def affichage_graph_reussite(X_test_scaled):
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

def estimation_prix_maison(nouvelle_maison, model, preprocessor):
    # Convertir le dictionnaire en DataFrame
    nouvelle_maison_df = pd.DataFrame([nouvelle_maison])

    # Appliquer le préprocesseur sur la nouvelle maison
    nouvelle_maison_encoded = preprocessor.transform(nouvelle_maison_df)

    prix_estime = model.predict(nouvelle_maison_encoded)
    return prix_estime[0]

#######################################2. Nettoyage#################################################################

#######################################3. Analyse#################################################################

#######################################4. Analyse#################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


X = df.drop(columns=['id', 'price', 'date', 'lat', 'long'])  # Caractéristiques (features), en excluant la colonne cible
y = df['price']  # Variable cible (target)

y_bins = pd.qcut(y, q=4, labels=False)

# Diviser les données en ensemble d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y_bins)


print("Ensemble d'entraînement :", X_train.shape)
print("Ensemble de test :", X_test.shape)


# Identifier les colonnes catégoriques
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
# OneHotEncoding pour les colonnes catégoriques
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ],
    remainder='passthrough'
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


#######################################5. Entraienemnt#################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train_encoded, y_train)

# Prédire les valeurs sur l'ensemble d'entraînement et de test
y_train_pred = model.predict(X_train_encoded)
y_test_pred = model.predict(X_test_encoded)

# Évaluer les performances du modèle
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Afficher les métriques de performance
print(f"Performances sur l'ensemble d'entraînement :")
print(f"  - RMSE : {train_rmse:.2f}")
print(f"  - R²   : {train_r2:.2f}")

print(f"\nPerformances sur l'ensemble de test :")
print(f"  - RMSE : {test_rmse:.2f}")
print(f"  - R²   : {test_r2:.2f}")


affichage_graph_reussite(X_test_encoded)
#affichage_pourcentage_reussite()



nouvelle_maison = {
    "bedrooms": 3,
    "bathrooms": 1,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "zipcode": "98178",
    "lat": 47.5112,
    "long": -122.257,
    "sqft_living15": 1340,
    "sqft_lot15": 5650
}#prix attendu 221900


# Estimation du prix
prix_estime = estimation_prix_maison(nouvelle_maison, model, preprocessor)
print(f"Le prix estimé de la maison est : ${prix_estime:,.2f} pour $221,900")
