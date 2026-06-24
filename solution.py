# -*- coding: utf-8 -*-

# === Importation des bibliothèques nécessaires ===
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier, cv

# === Définition des chemins et constantes ===
DOSSIER_DONNEES = Path("./titanic")
FICHIER_TRAIN = DOSSIER_DONNEES / "train.csv"
FICHIER_TEST = DOSSIER_DONNEES / "test.csv"
FICHIER_SUB_EXEMPLE = DOSSIER_DONNEES / "gender_submission.csv"
FICHIER_SORTIE = Path("./soumission_finale.csv")

COLS_CATEGORIELLES = [
    "Pclass", "SibSp", "Parch", "Ticket", "Cabin",
    "Embarked", "Titre", "Nb_mots_nom", "Est_enfant", "Sex"
]


# === Fonction de préparation des données ===
def preparer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les valeurs manquantes et ajoute des variables dérivées.
    """
    data = df.copy()

    # Gestion des valeurs manquantes
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Cabin"] = data["Cabin"].fillna("X").astype(str).str[0]
    data["Embarked"] = data["Embarked"].fillna("X")

    # Extraction du titre à partir du nom complet
    data["Titre"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Harmonisation des titres
    remplacements_miss = {
        "Mlle": "Miss", "Ms": "Miss", "Lady": "Miss", "Countess": "Miss",
        "Mme": "Miss", "Dona": "Miss"
    }
    remplacements_rares = {
        "Rev": "Rare", "Don": "Rare", "Capt": "Rare", "Major": "Rare",
        "Sir": "Rare", "Col": "Rare", "Jonkheer": "Rare"
    }

    data["Titre"] = data["Titre"].replace(remplacements_miss).replace(remplacements_rares)

    # Nombre de mots dans le nom
    data["Nb_mots_nom"] = data["Name"].str.split().str.len()

    # Indicateur enfant (booléen)
    data["Est_enfant"] = np.where(
        (data["Titre"].isin(["Master", "Miss"])) & (data["Age"] < 18),
        True,
        False
    )

    return data


# === Chargement des fichiers et préparation ===
def charger_donnees():
    train_df = pd.read_csv(FICHIER_TRAIN)
    test_df = pd.read_csv(FICHIER_TEST)
    _ = pd.read_csv(FICHIER_SUB_EXEMPLE)

    donnees_train = preparer_donnees(train_df)
    donnees_test = preparer_donnees(test_df)

    print("Données chargées et préparées avec succès.")
    print(f"Train : {donnees_train.shape} | Test : {donnees_test.shape}")
    return donnees_train, donnees_test


# === Entraînement du modèle CatBoost ===
def entrainer_modele(donnees_train: pd.DataFrame):
    X = donnees_train.drop(columns=["Survived", "PassengerId", "Name"])
    y = donnees_train["Survived"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.84, random_state=1
    )

    modele = CatBoostClassifier(
        eval_metric="Accuracy",
        use_best_model=True,
        random_seed=42
    )

    modele.fit(
        X_train, y_train,
        cat_features=COLS_CATEGORIELLES,
        eval_set=(X_valid, y_valid),
        silent=True
    )

    # Importance des variables
    pool_train = Pool(
        data=X_train,
        label=y_train,
        cat_features=COLS_CATEGORIELLES
    )
    importances = modele.get_feature_importance(pool_train)
    df_importances = (
        pd.DataFrame({"Variable": X_train.columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n📊 Top 10 des variables les plus importantes :")
    print(df_importances.head(10))

    return modele


# === Génération du fichier de soumission ===
def generer_soumission(modele: CatBoostClassifier, donnees_test: pd.DataFrame):
    X_test = donnees_test.drop(columns=["PassengerId", "Name"])
    predictions = modele.predict(X_test).astype(int)

    soumission = pd.DataFrame({
        "PassengerId": donnees_test["PassengerId"],
        "Survived": predictions
    })

    soumission.to_csv("./soumission_finale.csv", index=False)
    print("\n✅ Fichier 'soumission_finale.csv' créé avec succès !")


# === Programme principal ===
if __name__ == "__main__":
    donnees_entrainement, donnees_test = charger_donnees()
    modele_catboost = entrainer_modele(donnees_entrainement)
    generer_soumission(modele_catboost, donnees_test)
