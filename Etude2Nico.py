import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Chargement des données
df = pd.read_csv("C:/Users/nicol/Documents/GitHub/Projet_python_data/Engineering_graduate_salary.csv")

####################################### 1. Nettoyage des données #######################################
mean_salary = df['Salary'].mean()
df['Salary'] = df['Salary'].apply(lambda x: mean_salary if x > 1500000 else x)

####################################### 2. Préparation des données #######################################
# Suppression des colonnes inutiles
X = df.drop(columns=['Salary', '10percentage', '10board', '12graduation', '12percentage', '12board', 
                     'CollegeID', 'collegeGPA','ElectricalEngg', 'ElectronicsAndSemicon', 'ComputerProgramming', 
                     'MechanicalEngg', 'TelecomEngg', 'CollegeCityID', 'DOB', 'CollegeCityTier', 'ComputerScience', 'ID', 'CivilEngg'])
y = df['Salary']

# Création des catégories de salaire avec `qcut`
y_bins = pd.qcut(y, q=4, labels=False)

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_bins, test_size=0.2, random_state=42, stratify=y_bins)

# Colonnes catégoriques
categorical_columns = X.select_dtypes(include=['object']).columns

# Prétraitement : encodage des colonnes catégoriques et mise à l'échelle des colonnes numériques
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns)],
    remainder='passthrough'
)

# Encodage et mise à l'échelle
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

####################################### 3. Entraînement du modèle #######################################
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Prédictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

####################################### 4. Évaluation du modèle #######################################
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Précision sur l'ensemble d'entraînement : {train_accuracy * 100:.2f}%")
print(f"Précision sur l'ensemble de test : {test_accuracy * 100:.2f}%")

# Matrice de confusion
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_bins))
disp.plot(cmap='Blues')
plt.title('Matrice de confusion')
plt.show()

####################################### 5. Prédiction pour un nouvel employé #######################################
def predire_salaire(new_employee):
    new_data_df = pd.DataFrame(new_employee)
    new_data_encoded = preprocessor.transform(new_data_df)
    new_data_scaled = scaler.transform(new_data_encoded)
    predicted_category = model.predict(new_data_scaled)[0]
    print(f"Le salaire prédit pour cette personne appartient à la catégorie : {predicted_category}")

# Exemple de données pour un nouvel employé
new_employee = {
    "Gender": ["m"],
    "CollegeTier": [1],
    "Degree": ["B.Tech/B.E."],
    "Specialization": ["computer science & engineering"],
    "CollegeState": ["Uttar Pradesh"],
    "GraduationYear": [2014],
    "English": [440],
    "Logical": [435],
    "Quant": [210],
    "Domain": [0.342314899911815],
    "conscientiousness": [1.1336],
    "agreeableness": [0.0459],
    "extraversion": [1.2396],
    "nueroticism": [0.5262],
    "openess_to_experience": [-0.2859]
}

predire_salaire(new_employee)
