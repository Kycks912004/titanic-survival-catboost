# 🚢 Titanic Survival Prediction — CatBoost

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-FFCC00?style=for-the-badge&logo=yandex&logoColor=black)](https://catboost.ai/)
[![Accuracy](https://img.shields.io/badge/Accuracy-80.68%25-success?style=for-the-badge)](https://www.kaggle.com/c/titanic)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/titanic)

[![Solution](https://img.shields.io/badge/Code-solution.py-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/Kycks912004/titanic-survival-catboost/blob/main/solution.py)
[![Résultats](https://img.shields.io/badge/CSV-soumission__finale.csv-green?style=for-the-badge&logo=files&logoColor=white)](https://github.com/Kycks912004/titanic-survival-catboost/blob/main/soumission_finale.csv)
[![Train Data](https://img.shields.io/badge/Data-train.csv-orange?style=for-the-badge&logo=files&logoColor=white)](https://github.com/Kycks912004/titanic-survival-catboost/blob/main/titanic/train.csv)

> Prédiction de survie des passagers du Titanic avec **CatBoost** et feature engineering avancé.  
> **Précision obtenue : 80.68 %**

---

## 🏆 Résultats

| Métrique | Valeur |
|----------|--------|
| **Précision (validation)** | **80.68 %** |
| Algorithme | CatBoostClassifier |
| Split train/val | 84 % / 16 % |
| Random seed | 42 |

---

## 🧠 Approche

### Feature Engineering

| Feature créée | Description |
|--------------|-------------|
| `Titre` | Extrait du nom (`Mr`, `Miss`, `Master`, `Rare`…) |
| `Est_enfant` | `True` si Titre ∈ {Master, Miss} **et** Age < 18 |
| `Nb_mots_nom` | Nombre de mots dans le nom complet |
| `Cabin` | Première lettre de la cabine (`X` si manquant) |

### Gestion des valeurs manquantes

| Colonne | Stratégie |
|---------|-----------|
| `Age` | Médiane |
| `Fare` | Médiane |
| `Cabin` | `"X"` |
| `Embarked` | `"X"` |

### Modèle

- **CatBoostClassifier** avec gestion native des variables catégorielles
- Pas d'encodage manuel nécessaire (`cat_features` passé directement)
- `use_best_model=True` pour éviter l'overfitting

---

## 📁 Structure

```
titanic-survival-catboost/
├── solution.py              # Script principal (CatBoost)
├── requirements.txt         # Dépendances
├── soumission_finale.csv    # ✅ Résultats — prédictions finales
└── titanic/
    ├── train.csv            # Données d'entraînement (891 passagers)
    ├── test.csv             # Données de test (418 passagers)
    └── gender_submission.csv
```

## 📋 Aperçu des résultats (`soumission_finale.csv`)

| PassengerId | Survived |
|------------|----------|
| 892 | 0 |
| 893 | 0 |
| 894 | 0 |
| 895 | 0 |
| 896 | 1 |
| 897 | 0 |
| 898 | 1 |
| 899 | 0 |
| … | … |

→ [Voir le fichier complet](https://github.com/Kycks912004/titanic-survival-catboost/blob/main/soumission_finale.csv)

---

## 🚀 Lancer le projet

```bash
# 1. Cloner le repo
git clone https://github.com/Kycks912004/titanic-survival-catboost.git
cd titanic-survival-catboost

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger les données Kaggle et les placer dans titanic/
#    https://www.kaggle.com/c/titanic/data

# 4. Lancer le script
python solution.py
```

---

## 📊 Variables catégorielles utilisées

```python
COLS_CATEGORIELLES = [
    "Pclass", "SibSp", "Parch", "Ticket", "Cabin",
    "Embarked", "Titre", "Nb_mots_nom", "Est_enfant", "Sex"
]
```

---

## 👤 Auteur

**Kylian Pinto** — M1 Ingénierie Data & IA, ECE Paris  
[![GitHub](https://img.shields.io/badge/GitHub-Kycks912004-181717?style=flat&logo=github)](https://github.com/Kycks912004)
